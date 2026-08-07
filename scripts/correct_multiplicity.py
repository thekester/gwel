"""Apply family-wise error control to every paired claim the paper makes.

The paper reports a dozen paired differences with bootstrap intervals and, in
its limitations, concedes that no correction is applied. That concession is
cheap to replace with a result: reconstruct each comparison from the run
records, convert its bootstrap to a two-sided p-value, and run Holm's step-down
over the family.

The point is not to inflate the count of surviving claims. It is to say which
ones a reader should believe after accounting for how many questions were asked,
and to name the ones that do not survive rather than leaving them stated at
nominal significance.

Usage: PYTHONPATH=scripts python scripts/correct_multiplicity.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from gwel.config import load_config
from gwel.data.scoring import ScoringPolicy, rescore_records
from gwel.oracle.records import deduplicate_records, read_records
from gwel.oracle.token_cost import fit_token_cost
from gwel.router.decision import (
    escalation_delta,
    fit_correctness_rule,
    fit_gain_rule,
    fit_ladder_rule,
    signed_gain,
)
from gwel.router.multiplicity import bootstrap_p_value, family_wise_error, holm_bonferroni
from gwel.router.probes import fit_layer_probe
from gwel.router.splits import make_split

from evaluate_decision_rule import component_costs, load_arrays, policy_cost

MIXTURE = "configs/pilot1000.yaml"
SINGLE = "configs/docvqa1200.yaml"
LAYER = 6
MIN_BUCKET = 50


def mixture_tests() -> list[tuple[str, list[float]]]:
    """Paired difference vectors for the claims made on the four-dataset pilot."""
    config = load_config(MIXTURE)
    cheap_ok, full_ok, entropy, matrix, folds = load_arrays(
        config, "results/activations_full.npz", LAYER
    )
    train, test = folds["train"], folds["test"]
    gains = signed_gain(cheap_ok, full_ok)
    costs = component_costs()
    delta = escalation_delta(costs, read="probe")

    probe = fit_layer_probe(matrix[train], (gains[train] > 0).astype(float), LAYER)
    score = probe.score(matrix)
    tests: list[tuple[str, list[float]]] = []

    # D1: calibrating the gain against calibrating correctness, same signal.
    gain_rule = fit_gain_rule(
        score[train], gains[train], delta_ms=delta, value_ms_per_correct=800.0
    )
    ucci_rule = fit_correctness_rule(
        score[train], cheap_ok[train],
        full_accuracy=float(full_ok[train].mean()),
        delta_ms=delta, value_ms_per_correct=800.0,
    )
    g_fires, u_fires = gain_rule.escalate(score[test]), ucci_rule.escalate(score[test])
    tests.append((
        "D1 latency: gain rule under UCCI rule",
        (policy_cost(g_fires, costs, read="probe")
         - policy_cost(u_fires, costs, read="probe")).tolist(),
    ))

    # Probe policy against the entropy policy at matched escalation rates.
    for rate in (0.20, 0.30, 0.40):
        cut_e = np.quantile(entropy[train], 1 - rate)
        cut_p = np.quantile(score[train], 1 - rate)
        fires_e, fires_p = entropy[test] >= cut_e, score[test] >= cut_p
        tests.append((
            f"probe vs entropy latency @{rate:.0%}",
            (policy_cost(fires_p, costs, read="probe")
             - policy_cost(fires_e, costs, read="entropy")).tolist(),
        ))
        tests.append((
            f"probe vs entropy accuracy @{rate:.0%}",
            (np.where(fires_p, full_ok[test], cheap_ok[test]).astype(float)
             - np.where(fires_e, full_ok[test], cheap_ok[test]).astype(float)).tolist(),
        ))
    return tests


def single_domain_tests() -> list[tuple[str, list[float]]]:
    """Paired vectors for the claims made on the 1200-page DocVQA pilot."""
    config = load_config(SINGLE)
    grouped: dict[str, dict] = defaultdict(dict)
    for record in rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    ):
        grouped[record.example_id][record.config_id] = record
    cheap, rungs = "lowres_384", ("lowres_768", "lowres_1152", "full")
    ids = [
        e for e in grouped
        if all(c in grouped[e] for c in (cheap, *rungs)) and grouped[e][cheap].signals
    ]
    correct = {c: np.array([grouped[e][c].correct for e in ids]) for c in (cheap, *rungs)}
    tokens = {
        c: np.array([grouped[e][c].visual_tokens for e in ids], float)
        for c in (cheap, *rungs)
    }
    entropy = np.array([float(grouped[e][cheap].signals["mean_entropy"]) for e in ids])

    by = defaultdict(list)
    for e in ids:
        for r in grouped[e].values():
            by[int(r.visual_tokens)].append(r.latency_ms)
    good = [t for t in sorted(by) if len(by[t]) >= MIN_BUCKET]
    model = fit_token_cost(good, [float(np.median(by[t])) for t in good])
    latency = {c: model.predict(tokens[c]) for c in (cheap, *rungs)}
    gains = {r: signed_gain(correct[cheap], correct[r]) for r in rungs}
    deltas = np.column_stack([latency[r] - latency[cheap] for r in rungs])

    tests: list[tuple[str, list[float]]] = []
    # R1: each rung step, as a paired gain vector.
    for low, high in zip((cheap, *rungs)[:-1], rungs):
        vector = (correct[high] & ~correct[low]).astype(float) - (
            correct[low] & ~correct[high]
        ).astype(float)
        tests.append((f"R1 rung gain {low} to {high}", vector.tolist()))

    # R6: ladder against binary, at each operating point.
    rng = np.random.default_rng(5000)
    shuffled = rng.permutation(len(ids))
    test_index, train_index = shuffled[:300], shuffled[300:]
    for value in (800.0, 1600.0, 3200.0):
        fires = fit_gain_rule(
            entropy[train_index], gains["full"][train_index],
            delta_ms=float(deltas[train_index, -1].mean()), value_ms_per_correct=value,
        ).escalate(entropy[test_index])
        binary_ms = np.where(fires, latency["full"][test_index], latency[cheap][test_index])
        chosen = fit_ladder_rule(
            entropy[train_index], {r: gains[r][train_index] for r in rungs},
            value_ms_per_correct=value,
        ).choose(entropy[test_index], deltas[test_index])
        ladder_ms = np.where(chosen < 0, latency[cheap][test_index], 0.0)
        for level, rung in enumerate(rungs):
            ladder_ms = np.where(chosen == level, latency[rung][test_index], ladder_ms)
        tests.append((
            f"R6 ladder vs binary latency V={value:.0f}", (ladder_ms - binary_ms).tolist()
        ))
    return tests




def auroc_p_value(
    scores_a,
    scores_b,
    labels,
    *,
    resamples: int = 10000,
    seed: int = 1234,
) -> float:
    """Two-sided bootstrap p-value for a paired AUROC difference.

    AUROC is not a per-example mean, so the generic converter does not apply;
    examples are resampled jointly and the AUROC difference recomputed, which
    preserves the pairing between the two scores on each draw.
    """
    from gwel.router.evaluate import auroc

    scores_a = np.asarray(scores_a, dtype=np.float64)
    scores_b = np.asarray(scores_b, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    rng = np.random.default_rng(seed)
    n = len(labels)
    observed = auroc(scores_a.tolist(), [bool(x) for x in labels]) - auroc(
        scores_b.tolist(), [bool(x) for x in labels]
    )
    if observed == 0.0:
        return 1.0
    wrong = 0
    draws = 0
    for _ in range(resamples):
        idx = rng.integers(0, n, n)
        lab = labels[idx]
        if lab.all() or not lab.any():
            continue
        delta = auroc(scores_a[idx].tolist(), [bool(x) for x in lab]) - auroc(
            scores_b[idx].tolist(), [bool(x) for x in lab]
        )
        draws += 1
        if (delta <= 0) if observed > 0 else (delta >= 0):
            wrong += 1
    return float(min(1.0, max(2.0 * wrong / max(draws, 1), 1.0 / max(draws, 1))))


def headline_auroc_test():
    """The abstract's most visible number, previously outside the family.

    Probe against entropy on the conditional recovery target, over the
    test-fold failures: the +0.143 [+0.031, +0.264] claim of the abstract.
    """
    config = load_config(MIXTURE)
    cheap_ok, full_ok, entropy, matrix, folds = load_arrays(
        config, "results/activations_full.npz", LAYER
    )
    train, test = folds["train"], folds["test"]
    train_failures = train[~cheap_ok[train]]
    test_failures = test[~cheap_ok[test]]
    probe = fit_layer_probe(
        matrix[train_failures], full_ok[train_failures].astype(float), LAYER
    )
    p = auroc_p_value(
        probe.score(matrix[test_failures]),
        entropy[test_failures],
        full_ok[test_failures],
    )
    return ("P1 AUROC: probe vs entropy on the recovery target", p)


def review_tests() -> list[tuple[str, list[float]]]:
    """Paired comparisons added after the first multiplicity pass.

    Each artefact stores the per-resample (or per-example) vector behind its
    interval, so these are scored by the same bootstrap as the rest of the
    family rather than by an approximation to a reported width. An earlier
    version reconstructed a p-value from the interval instead; it is kept out
    of the repository because it treated a sampling distribution as a set of
    observations and returned the resolution floor for every test.

    Only comparisons the text leans on as evidence enter the family. Nulls are
    included too, since excluding them would make the correction selective.
    """
    tests: list[tuple[str, list[float]]] = []

    def add(name: str, vector) -> None:
        if vector:
            tests.append((name, [float(v) for v in vector]))

    free = json.loads(Path("results/free_signal.json").read_text())
    for costing in ("flat", "per-example"):
        for name, row in free[costing]["preference swept"].items():
            if name.startswith("image size"):
                add(f"CV2 free signal clears the hull, {costing}, {name}",
                    row.get("gap_vector"))
        for key, row in free[costing]["free minus probe"].items():
            if key.endswith("vector"):
                add(f"CV2 free signal minus probe, {costing}, {key[:-7]}", row)

    if Path("results/free_signal_256m.json").exists():
        small = json.loads(Path("results/free_signal_256m.json").read_text())
        for costing in ("flat", "per-example"):
            for name, row in small[costing]["preference swept"].items():
                if name.startswith(("image size", "probe")):
                    add(f"CV6 256M hull gap, {costing}, {name}", row.get("gap_vector"))

    domain = json.loads(Path("results/domain_policy.json").read_text())
    for name, row in domain["policies"].items():
        if name.startswith("domain label"):
            add(f"CV3 domain label clears the hull, {name}", row.get("gap_vector"))

    for label, path in (
        ("DocVQA", "results/free_signal_docvqa.json"),
        ("InfographicVQA", "results/free_signal_infovqa.json"),
        ("ChartQA", "results/free_signal_chartqa.json"),
        ("DocVQA 256M", "results/free_signal_docvqa_256m.json"),
        ("DocVQA LLaVA-OV", "results/free_signal_llavaov.json"),
        ("DocVQA Qwen2-VL-2B", "results/free_signal_qwen2b.json"),
        ("ChartQA LLaVA-OV", "results/free_signal_chartqa_llavaov.json"),
        ("TextVQA", "results/free_signal_textvqa.json"),
        ("DocVQA SmolVLM2-2.2B", "results/free_signal_docvqa_2b.json"),
        ("InfoVQA Qwen2-VL-2B", "results/free_signal_infovqa_qwen2b.json"),
    ):
        if not Path(path).exists():
            continue
        for name, row in json.loads(Path(path).read_text()).items():
            add(f"CV4 {label} hull gap, {name}", row.get("gap_vector"))

    tile = json.loads(Path("results/tile_budget_analysis.json").read_text())
    for step in tile["steps"]:
        for key in ("all", "tokens_rose"):
            block = step.get(key)
            if block:
                add(f"R13 tile budget {step['from']} to {step['to']} ({key})",
                    block.get("vector"))

    corpora = json.loads(Path("results/corpus_ceilings.json").read_text())
    for run in corpora["runs"]:
        for step in run["steps"]:
            add(
                f"R14 {run['label']} rung gain {step['from']} to {step['to']}",
                step.get("vector"),
            )

    budget = json.loads(Path("results/fixed_budget.json").read_text())
    for step in budget["steps"]:
        add(f"R12 fixed budget {step['from']} to {step['to']}", step.get("vector"))

    # The comparator's own slack. These enter the family because the paper
    # leans on two of them as retractions and on the rest as the bar its
    # surviving gaps clear, which is evidentiary use either way.
    # The equivalence comparison. Its per-example paired differences are
    # observations, not bootstrap replicates, so the family's bootstrap applies
    # to them directly.
    equivalence = Path("results/equivalence.json")
    if equivalence.exists():
        row = json.loads(equivalence.read_text())
        vector = row.get("difference_vector")
        if vector:
            add("D2 accuracy: gain rule under the error-probability rule", vector)

    second = Path("results/second_mixture.json")
    if second.exists():
        for name, row in json.loads(second.read_text())["hull_gaps"].items():
            add(f"CV20 second mixture, {name}", row.get("gap_vector"))

    retimed = Path("results/oracle_slack_retimed.json")
    if retimed.exists():
        rows = json.loads(retimed.read_text())
        for key in ("oracle_single_shot", "oracle_averaged", "deployable_tokens"):
            add(f"CV17 re-timed slack, {key}", rows[key].get("gap_vector"))

    for tag, slack_path in (
        ("binary", Path("results/cost_only.json")),
        ("graded", Path("results/cost_only_graded.json")),
    ):
        if slack_path.exists():
            for label, row in json.loads(slack_path.read_text()).items():
                add(f"CV12 cost-only slack, {tag}, {label}", row.get("gap_vector"))


    # The four refuted accounts of the descriptor's residual (CV14) are
    # deliberately NOT in this family. Two reasons. Their artefact stores
    # bootstrap replicates of an AUROC, and `bootstrap_p_value` expects paired
    # differences over observations: resampling replicates would shrink the
    # standard error by the square root of their count and declare every one of
    # them significant. And the text leans on them to say that nothing
    # separates, which is a claim about intervals covering 0.5 rather than a
    # comparison; entering thirty-two non-claims here would also inflate the
    # correction against the results we do assert. They are reported with
    # confidence intervals in the paper instead.
    return tests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--out", default="results/multiplicity.json")
    args = parser.parse_args()

    family: list[tuple[str, list[float]]] = []
    for source in (mixture_tests, single_domain_tests, review_tests):
        try:
            family.extend(source())
        except FileNotFoundError as error:
            print(f"skipping {source.__name__}: missing {error.filename}")

    scored = [(name, bootstrap_p_value(values)) for name, values in family]
    try:
        scored.append(headline_auroc_test())
    except FileNotFoundError as error:
        print(f"skipping headline AUROC: missing {error.filename}")
    corrected = holm_bonferroni(scored, alpha=args.alpha)

    print(
        f"{len(scored)} paired tests. Uncorrected, the chance of at least one "
        f"false positive is {family_wise_error(len(scored), alpha=args.alpha):.0%}.\n"
    )
    width = max(len(name) for name, _ in scored) + 2
    print(f"{'test':<{width}}{'p':>10}{'Holm p':>10}{'survives':>10}")
    for row in sorted(corrected, key=lambda r: r.p_value):
        print(
            f"{row.name:<{width}}{row.p_value:>10.4f}{row.adjusted:>10.4f}"
            f"{('yes' if row.survives else 'no'):>10}"
        )

    survivors = [r for r in corrected if r.survives]
    nominal = [r for r in corrected if r.p_value <= args.alpha]
    print(
        f"\n{len(nominal)}/{len(corrected)} clear the nominal level; "
        f"{len(survivors)}/{len(corrected)} survive Holm at alpha={args.alpha}"
    )
    lost = [r.name for r in corrected if r.p_value <= args.alpha and not r.survives]
    if lost:
        print("lost to the correction: " + "; ".join(lost))

    Path(args.out).write_text(
        json.dumps(
            {
                "alpha": args.alpha,
                "tests": [
                    {
                        "name": r.name,
                        "p_value": r.p_value,
                        "adjusted": r.adjusted,
                        "survives": r.survives,
                    }
                    for r in corrected
                ],
                "nominal": len(nominal),
                "survivors": len(survivors),
                "family_wise_error_uncorrected": family_wise_error(
                    len(corrected), alpha=args.alpha
                ),
            },
            indent=2,
        )
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
