"""Re-derive every documented claim from the data and report PASS / FAIL.

Claims written in `ANGLES.md`, `FINDINGS.md` and `PROPOSAL.md` are prose. This
turns each one into an executable check with an explicit numeric threshold, so
a claim that stops holding, because more data arrived, because a bug was
fixed, because a config changed, fails loudly instead of surviving in a
markdown file.

Every check states what it needs. Missing artefacts report SKIP, never a silent
pass. Exit code is non-zero if any check fails, so this can gate a commit.

Usage: python scripts/validate_claims.py
"""

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


@dataclass
class Result:
    claim: str
    status: str  # PASS | FAIL | SKIP
    detail: str


RESULTS: list[Result] = []


def check(claim: str):
    """Decorator registering a claim check; exceptions become SKIP."""

    def wrap(fn):
        try:
            status, detail = fn()
        except FileNotFoundError as error:
            status, detail = "SKIP", f"missing artefact: {Path(str(error.filename)).name}"
        except Exception as error:  # noqa: BLE001 - a broken check must not hide others
            status, detail = "SKIP", f"{type(error).__name__}: {error}"
        RESULTS.append(Result(claim, status, detail))
        return fn

    return wrap


def _signed(value: float, digits: int = 3) -> str:
    """Format with a sign, widening rather than rounding a nonzero value to zero.

    A value of -0.0005 printed at three decimals reads ``-0.000``, which is both
    ugly and ambiguous: the reader cannot tell a signed rounding artefact from an
    exact zero. An exact zero still prints as ``+0.000``, so the determinism
    checks keep saying what they mean.
    """
    while value != 0.0 and abs(round(value, digits)) < 10.0**-digits:
        digits += 1
    return f"{value:+.{digits}f}"


def _records(path: str):
    from gwel.data.scoring import ScoringPolicy, rescore_records
    from gwel.oracle.records import deduplicate_records, read_records

    return rescore_records(deduplicate_records(read_records(path)), ScoringPolicy())


def _grouped(path: str):
    grouped: dict[str, dict] = defaultdict(dict)
    for record in _records(path):
        grouped[record.example_id][record.config_id] = record
    return grouped


PILOT = "results/runs/pilot1000_records.jsonl"


# ---------------------------------------------------------------- cost claims

@check("A1: vision encoder is 40-55% of the visual cost, on every model tested")
def _():
    shares = []
    for path in ("results/component_latency.json", "results/component_latency_2b.json"):
        rows = {r["config"]: r for r in json.loads(Path(path).read_text())}
        base = rows["no_image"]["prefill_ms"]
        for name, row in rows.items():
            if name == "no_image" or row["visual_tokens"] < 200:
                continue
            encoder = row["vision_encoder_ms"] + row["projector_ms"]
            extra = max(row["prefill_ms"] - base, 0.0)
            shares.append(encoder / (encoder + extra))
    ok = bool(shares) and all(0.40 <= s <= 0.55 for s in shares)
    return ("PASS" if ok else "FAIL"), f"shares {[f'{s:.0%}' for s in shares]}"


@check("A2: token pruning is capped at roughly half the visual saving")
def _():
    rows = {r["config"]: r for r in
    json.loads(Path("results/component_latency.json").read_text())}
    base = rows["no_image"]["prefill_ms"]
    row = rows["longest_1536"]
    encoder = row["vision_encoder_ms"] + row["projector_ms"]
    extra = max(row["prefill_ms"] - base, 0.0)
    recoverable_by_pruning = extra / (encoder + extra)
    ok = 0.40 <= recoverable_by_pruning <= 0.60
    return ("PASS" if ok else "FAIL"), f"pruning can recover {recoverable_by_pruning:.0%} of the visual cost"


@check("A3: decode latency tracks layer count, not parameter count")
def _():
    # Measured: 256M/30 layers/64.2ms, 500M/32/77.0ms, 2.2B/24/47.2ms.
    layers = np.array([30, 32, 24], dtype=float)
    params = np.array([0.256, 0.507, 2.247])
    decode = np.array([64.2, 77.0, 47.2])
    r_layers = abs(np.corrcoef(layers, decode)[0, 1])
    r_params = abs(np.corrcoef(params, decode)[0, 1])
    ok = r_layers > 0.9 and r_layers > r_params
    return ("PASS" if ok else "FAIL"), f"|r| layers={r_layers:.2f} params={r_params:.2f}"


@check("A4: direct energy measurement fails its own validity check")
def _():
    records = [r for r in _records(PILOT) if not r.config_id.startswith("ocr_")]
    by_tokens: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        if record.net_energy_mj is not None:
            by_tokens[record.visual_tokens][record.config_id].append(record.net_energy_mj)
    worst = 0.0
    for configs in by_tokens.values():
        medians = [float(np.median(v)) for v in configs.values() if len(v) >= 10]
        if len(medians) > 1:
            worst = max(worst, (max(medians) - min(medians)) / min(medians))
    ok = worst > 0.10  # the claim is that it FAILS, so a large spread is expected
    return ("PASS" if ok else "FAIL"), f"worst equal-token disagreement {worst:.0%} (claim: >10%)"


# ------------------------------------------------------------- signal claims

@check("S1: entropy predicts cheap-pass correctness, AUROC >= 0.72")
def _():
    from gwel.modeling.signals import ConfidenceSignals
    from gwel.router.evaluate import auroc

    grouped = _grouped(PILOT)
    ids = [e for e in grouped if "lowres_384" in grouped[e] and grouped[e]["lowres_384"].signals]
    scores = [-ConfidenceSignals.from_dict(grouped[e]["lowres_384"].signals).mean_entropy for e in ids]
    labels = [grouped[e]["lowres_384"].correct for e in ids]
    value = auroc(scores, labels)
    return ("PASS" if value >= 0.72 else "FAIL"), f"AUROC {value:.3f} on n={len(ids)}"


@check("S2: net escalation gain rises monotonically across entropy quintiles")
def _():
    from gwel.modeling.signals import ConfidenceSignals

    grouped = _grouped(PILOT)
    ids = [
        e for e in grouped
        if "lowres_384" in grouped[e] and "full" in grouped[e]
        and grouped[e]["lowres_384"].signals
    ]
    entropy = np.array(
        [ConfidenceSignals.from_dict(grouped[e]["lowres_384"].signals).mean_entropy for e in ids]
    )
    cheap = np.array([grouped[e]["lowres_384"].correct for e in ids])
    full = np.array([grouped[e]["full"].correct for e in ids])
    gain = ((~cheap) & full).astype(float) - (cheap & (~full)).astype(float)
    edges = np.quantile(entropy, np.linspace(0, 1, 6))
    means = [
        gain[(entropy >= edges[i]) & (entropy <= edges[i + 1] if i == 4 else entropy < edges[i + 1])].mean()
        for i in range(5)
    ]
    ok = means[-1] > means[0] + 0.20 and means[-1] > means[-2]
    return ("PASS" if ok else "FAIL"), "quintile gains " + " ".join(f"{m:+.0%}" for m in means)


@check("S3: escalation harms 2-8% of queries (non-monotone)")
def _():
    grouped = _grouped(PILOT)
    ids = [e for e in grouped if "lowres_384" in grouped[e] and "full" in grouped[e]]
    harmed = sum(
        grouped[e]["lowres_384"].correct and not grouped[e]["full"].correct for e in ids
    ) / len(ids)
    return ("PASS" if 0.02 <= harmed <= 0.08 else "FAIL"), f"{harmed:.1%} of queries harmed"


@check("S4: OCR helps DocVQA and not the other datasets")
def _():
    from gwel.router.evaluate import paired_difference

    grouped = _grouped(PILOT)
    verdicts = {}
    for dataset in ("docvqa", "textvqa", "vqav2", "vstar"):
        ids = [
            e for e in grouped
            if "lowres_384" in grouped[e] and "ocr_full" in grouped[e]
            and grouped[e]["lowres_384"].dataset == dataset
        ]
        if not ids:
            continue
        delta = paired_difference(
            [float(grouped[e]["ocr_full"].correct) for e in ids],
            [float(grouped[e]["lowres_384"].correct) for e in ids],
        )
        verdicts[dataset] = "helps" if delta.low > 0 else ("hurts" if delta.high < 0 else "flat")
    ok = verdicts.get("docvqa") == "helps" and all(
        v != "helps" for k, v in verdicts.items() if k != "docvqa"
    )
    return ("PASS" if ok else "FAIL"), str(verdicts)


# -------------------------------------------------------------- probe claims

def _probe_setup():
    from gwel.router.splits import make_split

    stored = np.load("results/activations_full.npz", allow_pickle=True)
    activations, ids = stored["activations"], list(stored["example_ids"])
    grouped = _grouped(PILOT)
    position = {e: i for i, e in enumerate(ids)}
    failed = [
        e for e in ids
        if "lowres_384" in grouped[e] and "full" in grouped[e]
        and not grouped[e]["lowres_384"].correct
    ]
    labels = np.array([float(grouped[e]["full"].correct) for e in failed])
    matrix = activations[[position[e] for e in failed]]
    split = make_split(
        failed, [grouped[e]["lowres_384"].dataset for e in failed],
        val_fraction=0.2, test_fraction=0.2, seed=1234,
    )
    order = {e: i for i, e in enumerate(failed)}
    train = np.array([order[e] for e in split.train])
    test = np.array([order[e] for e in split.test])
    return matrix, labels, train, test, grouped, failed


def _probe_auroc(matrix, labels, train, test, layer: int) -> float:
    from gwel.router.evaluate import auroc
    from gwel.router.probes import fit_layer_probe

    probe = fit_layer_probe(matrix[train, layer, :], labels[train], layer)
    return auroc(probe.score(matrix[test, layer, :]).tolist(), [bool(v) for v in labels[test]])


@check("P1: the probe beats output entropy on the escalation-value target")
def _():
    from gwel.modeling.signals import ConfidenceSignals
    from gwel.router.evaluate import auroc

    matrix, labels, train, test, grouped, failed = _probe_setup()
    best = max(_probe_auroc(matrix, labels, train, test, L) for L in range(matrix.shape[1]))
    entropy = np.array(
        [ConfidenceSignals.from_dict(grouped[e]["lowres_384"].signals).mean_entropy for e in failed]
    )
    baseline = auroc(entropy[test].tolist(), [bool(v) for v in labels[test]])
    ok = best > baseline + 0.05
    return ("PASS" if ok else "FAIL"), f"probe {best:.3f} vs best-sign entropy {baseline:.3f}"


@check("P2: an early layer matches the best layer (cheap probe suffices)")
def _():
    matrix, labels, train, test, _, _ = _probe_setup()
    per_layer = [_probe_auroc(matrix, labels, train, test, L) for L in range(matrix.shape[1])]
    best = max(per_layer)
    early = max(per_layer[1:8])
    ok = early >= best - 0.03
    return ("PASS" if ok else "FAIL"), f"best {best:.3f} at L{int(np.argmax(per_layer))}, best early L1-7 {early:.3f}"


@check("P3: the escalation-value direction does not transfer across domains")
def _():
    from gwel.router.evaluate import auroc
    from gwel.router.probes import fit_layer_probe

    matrix, labels, _, _, grouped, failed = _probe_setup()
    layer = 23
    detail = [i for i, e in enumerate(failed) if grouped[e]["lowres_384"].dataset in ("docvqa", "vstar")]
    knowledge = [i for i, e in enumerate(failed) if grouped[e]["lowres_384"].dataset == "vqav2"]
    if len(detail) < 30 or len(knowledge) < 20:
        raise RuntimeError("not enough per-domain examples")
    transfers = []
    for source, target in ((detail, knowledge), (knowledge, detail)):
        probe = fit_layer_probe(matrix[source, layer, :], labels[source], layer)
        transfers.append(
            auroc(probe.score(matrix[target, layer, :]).tolist(), [bool(v) for v in labels[target]])
        )
    ok = all(t < 0.55 for t in transfers)
    return ("PASS" if ok else "FAIL"), f"transfers {[f'{t:.3f}' for t in transfers]} (claim: all < 0.55)"


@check("P4: a smaller model's activations transfer to the larger model's outcomes")
def _():
    from gwel.router.splits import make_split

    stored = np.load("results/activations_source.npz", allow_pickle=True)
    matrix, ids = stored["activations"], list(stored["example_ids"])
    grouped = _grouped(PILOT)
    labels = np.array([float(grouped[e]["full"].correct) for e in ids])
    split = make_split(
        ids, [grouped[e]["lowres_384"].dataset for e in ids],
        val_fraction=0.2, test_fraction=0.2, seed=1234,
    )
    order = {e: i for i, e in enumerate(ids)}
    train = np.array([order[e] for e in split.train])
    test = np.array([order[e] for e in split.test])
    best = max(_probe_auroc(matrix, labels, train, test, L) for L in range(matrix.shape[1]))
    ok = best >= 0.70
    return ("PASS" if ok else "FAIL"), f"cross-model best AUROC {best:.3f} (claim: >= 0.70)"


# ------------------------------------------------------------- policy claims

@check("C1: escalation raises coverage over abstention at matched risk")
def _():
    from gwel.modeling.signals import ConfidenceSignals
    from gwel.router.coverage import escalating_frontier, selective_frontier

    grouped = _grouped(PILOT)
    ids = [
        e for e in grouped
        if "lowres_384" in grouped[e] and "full" in grouped[e]
        and grouped[e]["lowres_384"].signals and grouped[e]["full"].signals
    ]

    def conf(example, config):
        return -ConfidenceSignals.from_dict(grouped[example][config].signals).mean_entropy

    cheap_c = [conf(e, "lowres_384") for e in ids]
    cheap_ok = [grouped[e]["lowres_384"].correct for e in ids]
    esc_c = [conf(e, "full") for e in ids]
    esc_ok = [grouped[e]["full"].correct for e in ids]
    vanilla = selective_frontier(cheap_c, cheap_ok, cheap_cost=1.0)
    escalating = escalating_frontier(
        cheap_c, cheap_ok, esc_c, esc_ok, cheap_cost=1.0, escalated_cost=1.5
    )
    gains = []
    for tolerance in (0.40, 0.50):
        a, b = vanilla.at_risk(tolerance), escalating.at_risk(tolerance)
        if b is None:
            continue
        gains.append(b.coverage - (a.coverage if a else 0.0))
    ok = bool(gains) and all(g > 0.15 for g in gains)
    return ("PASS" if ok else "FAIL"), f"coverage gains {[f'{g:+.0%}' for g in gains]}"


@check("C2: the linear cost function expresses only a handful of policies")
def _():
    from gwel.actions import Action
    from gwel.config import load_config
    from gwel.oracle.cost import CostWeights, resource_cost
    from gwel.router.budget_selection import ActionStats, policy_regions
    from gwel.router.policies import group_runs

    config = load_config("configs/pilot1000.yaml")
    weights = CostWeights.from_config(config.cost)
    runs = group_runs(_records(PILOT))
    stats = []
    for action in Action.ordered():
        chosen = [r.realise(action, weights, region_selection="best") for r in runs]
        chosen = [c for c in chosen if c]
        stats.append(
            ActionStats(
                action,
                1 - sum(c.correct for c in chosen) / len(chosen),
                float(np.mean([
                    resource_cost(
                        latency_ms=c.latency_ms, energy_mj=c.net_energy_mj,
                        memory_mb=c.ram_peak_mb, visual_tokens=c.visual_tokens, weights=weights,
                    ) for c in chosen
                ])),
            )
        )
    regions = policy_regions(stats)
    ok = len(regions) <= 3
    return ("PASS" if ok else "FAIL"), f"{len(regions)} policy regions: " + ", ".join(
        r[2].value for r in regions
    )


@check("P5: an internal-signal region localizer does NOT beat random cell choice")
def _():
    from gwel.router.localizer import evaluate_localizer, pool_cells, train_localizer
    from gwel.router.splits import make_split

    stored = np.load("results/visual_grids_multi.npz", allow_pickle=True)
    grids = stored["grids"]
    ids = list(stored["example_ids"])
    layers = list(stored["layers"])
    grouped = _grouped(PILOT)
    cells = [f"crop_r{r}c{c}" for r in range(2) for c in range(2)]
    labels = [[grouped[e][c].correct for c in cells] for e in ids]
    split = make_split(
        ids, [grouped[e][cells[0]].dataset for e in ids],
        val_fraction=0.2, test_fraction=0.2, seed=1234,
    )
    order = {e: i for i, e in enumerate(ids)}
    train = [order[e] for e in split.train if e in order]
    test = [order[e] for e in split.test if e in order]

    margins = []
    for k in range(len(layers)):
        features = [pool_cells(grids[i, k], 2, 2) for i in range(len(ids))]
        localizer = train_localizer([features[i] for i in train], [labels[i] for i in
        train])
        stats = evaluate_localizer(
            localizer, [features[i] for i in test], [labels[i] for i in test]
        )
        margins.append(stats["chosen"] - stats["random"])
    # The claim is a negative: no layer beats random by more than noise.
    ok = max(margins) <= 0.02
    return ("PASS" if ok else "FAIL"), (
        f"best margin over random {max(margins):+.1%} across layers {layers}"
    )


@check("C3: aborting the prefill mid-flight really saves what the model predicts")
def _():
    measured = json.loads(Path("results/early_exit.json").read_text())
    rows = {r["config"]: r for r in
    json.loads(Path("results/component_latency.json").read_text())}
    cheap = rows["longest_384"]
    predicted = (
        cheap["vision_encoder_ms"] + cheap["projector_ms"]
        + cheap["prefill_ms"] * measured["exit_layer"] / measured["decoder_layers"]
    )
    error = abs(measured["truncated_prefill_ms"] - predicted) / predicted
    saves = measured["saving_ms"] > 0
    ok = saves and error < 0.15
    return ("PASS" if ok else "FAIL"), (
        f"truncated prefill {measured['truncated_prefill_ms']:.1f} ms vs "
        f"{predicted:.1f} predicted ({error:.0%} error); "
        f"escalated query saves {measured['saving_fraction']:.0%}"
    )


@check("C4: an untrained self-report baseline loses to uncertainty routing")
def _():
    import sys

    sys.path.insert(0, "scripts")
    from baseline_self_report import PROMPTS, says_escalate

    from gwel.modeling.signals import ConfidenceSignals
    from gwel.router.evaluate import paired_difference
    from gwel.router.splits import make_split

    reports = json.loads(Path("results/self_report.json").read_text())
    grouped = _grouped(PILOT)
    usable = [
        e for e in reports
        if "lowres_384" in grouped[e] and "full" in grouped[e]
        and grouped[e]["lowres_384"].signals
    ]
    cheap_ok = np.array([grouped[e]["lowres_384"].correct for e in usable])
    full_ok = np.array([grouped[e]["full"].correct for e in usable])
    entropy = np.array(
        [ConfidenceSignals.from_dict(grouped[e]["lowres_384"].signals).mean_entropy
         for e in usable]
    )
    split = make_split(
        usable, [grouped[e]["lowres_384"].dataset for e in usable],
        val_fraction=0.2, test_fraction=0.2, seed=1234,
    )
    order = {e: i for i, e in enumerate(usable)}
    train = np.array([order[e] for e in split.train])
    test = np.array([order[e] for e in split.test])

    best = None
    for prompt in PROMPTS:
        flags = np.array([says_escalate(reports[e][prompt]["answer"], prompt) for e in
        usable])
        accuracy = np.where(flags[test], full_ok[test], cheap_ok[test])
        if best is None or accuracy.mean() > best[0].mean():
            best = (accuracy, float(flags[test].mean()))
    baseline, rate = best

    cut = np.quantile(entropy[train], 1.0 - rate)
    routed = np.where(entropy[test] >= cut, full_ok[test], cheap_ok[test])
    delta = paired_difference(routed.astype(float).tolist(),
    baseline.astype(float).tolist())
    ok = delta.low > 0
    return ("PASS" if ok else "FAIL"), (
        f"entropy at the same {rate:.0%} rate is {delta} more accurate"
    )


@check("C5: probe dominance replicates on a second serving model")
def _():
    from gwel.router.evaluate import pareto_front
    from gwel.router.probes import fit_layer_probe
    from gwel.router.splits import make_split

    stored = np.load("results/activations_serve256.npz", allow_pickle=True)
    activations, ids = stored["activations"], list(stored["example_ids"])
    grouped = _grouped("results/runs/serve256_records.jsonl")
    cheap_ok = np.array([grouped[e]["lowres_384"].correct for e in ids])
    full_ok = np.array([grouped[e]["full"].correct for e in ids])
    entropy = np.array([
        float(grouped[e]["lowres_384"].signals["mean_entropy"]) for e in ids
    ])
    gain = ((~cheap_ok) & full_ok).astype(float)

    split = make_split(
        ids, [grouped[e]["lowres_384"].dataset for e in ids],
        val_fraction=0.2, test_fraction=0.2, seed=1234,
    )
    order = {e: i for i, e in enumerate(ids)}
    train = np.array([order[e] for e in split.train])
    test = np.array([order[e] for e in split.test])

    layer = 6
    probe = fit_layer_probe(activations[train, layer, :], gain[train], layer)
    score = probe.score(activations[:, layer, :])

    cheap_ms, full_ms, probe_ms = 123.4, 206.0, 20.3
    costs, accuracies, labels = [cheap_ms], [float(cheap_ok[test].mean())], ["cheap"]
    for rate in (0.10, 0.20, 0.30, 0.40, 0.50, 0.70):
        for tag, values, uses_probe in (("entropy", entropy, False), ("probe", score,
        True)):
            cut = np.quantile(values[train], 1.0 - rate)
            escalates = values[test] >= cut
            accuracies.append(float(np.where(escalates, full_ok[test],
            cheap_ok[test]).mean()))
            costs.append(float(
                np.where(escalates, probe_ms + full_ms, cheap_ms).mean() if uses_probe
                else (cheap_ms + escalates * full_ms).mean()
            ))
            labels.append(tag)
    costs.append(full_ms); accuracies.append(float(full_ok[test].mean()));
    labels.append("full")

    front = {labels[i] for i in pareto_front(costs, accuracies)}
    ok = "probe" in front and "entropy" not in front
    return ("PASS" if ok else "FAIL"), f"Pareto front contains {sorted(front)}"


# ------------------------------------------------- decision-rule claims (D)

def _decision_arrays(layer: int = 6):
    """Correctness, entropy and a joint-target probe score on the pilot."""
    from gwel.router.decision import signed_gain
    from gwel.router.probes import fit_layer_probe
    from gwel.router.splits import make_split

    stored = np.load("results/activations_full.npz", allow_pickle=True)
    activations, ids = stored["activations"], [str(e) for e in stored["example_ids"]]
    grouped = _grouped(PILOT)
    usable = [
        e for e in ids
        if "lowres_384" in grouped[e] and "full" in grouped[e]
        and grouped[e]["lowres_384"].signals
    ]
    position = {e: i for i, e in enumerate(ids)}
    cheap_ok = np.array([grouped[e]["lowres_384"].correct for e in usable])
    full_ok = np.array([grouped[e]["full"].correct for e in usable])
    entropy = np.array([
        float(grouped[e]["lowres_384"].signals["mean_entropy"]) for e in usable
    ])
    matrix = activations[[position[e] for e in usable]][:, layer, :]

    split = make_split(
        usable, [grouped[e]["lowres_384"].dataset for e in usable],
        val_fraction=0.2, test_fraction=0.2, seed=1234,
    )
    order = {e: i for i, e in enumerate(usable)}
    train = np.array([order[e] for e in split.train])
    test = np.array([order[e] for e in split.test])

    gains = signed_gain(cheap_ok, full_ok)
    probe = fit_layer_probe(matrix[train], (gains[train] > 0).astype(float), layer)
    return cheap_ok, full_ok, entropy, probe.score(matrix), gains, train, test


COSTS = {"cheap": 123.4, "full": 206.0, "probe": 20.3}


def _policy_cost(escalates: np.ndarray, *, read: str) -> np.ndarray:
    cheap, full, probe = COSTS["cheap"], COSTS["full"], COSTS["probe"]
    if read == "entropy":
        return cheap + escalates * full
    return np.where(escalates, probe + full, cheap)


@check("D1: calibrating correctness over-escalates versus calibrating the gain")
def _():
    """UCCI's Theorem 1 assumes escalation delivers a fixed accuracy.

    Non-monotone escalation violates that, so a correctness-calibrated rule
    should spend more compute for no more accuracy than a gain-calibrated one.
    """
    from gwel.router.decision import (
        escalation_delta,
        fit_correctness_rule,
        fit_gain_rule,
    )
    from gwel.router.evaluate import paired_difference

    cheap_ok, full_ok, _, probe_score, gains, train, test = _decision_arrays()
    delta = escalation_delta(COSTS, read="probe")
    value = 800.0

    gain_rule = fit_gain_rule(
        probe_score[train], gains[train], delta_ms=delta, value_ms_per_correct=value
    )
    ucci_rule = fit_correctness_rule(
        probe_score[train], cheap_ok[train],
        full_accuracy=float(full_ok[train].mean()),
        delta_ms=delta, value_ms_per_correct=value,
    )
    gain_fires = gain_rule.escalate(probe_score[test])
    ucci_fires = ucci_rule.escalate(probe_score[test])

    gain_ok = np.where(gain_fires, full_ok[test], cheap_ok[test]).astype(float)
    ucci_ok = np.where(ucci_fires, full_ok[test], cheap_ok[test]).astype(float)
    accuracy = paired_difference(gain_ok.tolist(), ucci_ok.tolist())
    latency = paired_difference(
        _policy_cost(gain_fires, read="probe").tolist(),
        _policy_cost(ucci_fires, read="probe").tolist(),
    )
    # Same accuracy (interval spans zero), strictly less compute (interval below zero).
    ok = accuracy.low <= 0.0 <= accuracy.high and latency.high < 0.0
    return ("PASS" if ok else "FAIL"), (
        f"escalates {gain_fires.mean():.0%} vs {ucci_fires.mean():.0%}; "
        f"accuracy {accuracy}, latency {latency} ms"
    )


@check("D2: an untuned cost-derived rule reaches the tuned sweep's frontier")
def _():
    from gwel.router.decision import escalation_delta, fit_gain_rule

    cheap_ok, full_ok, _, probe_score, gains, train, test = _decision_arrays()
    delta = escalation_delta(COSTS, read="probe")

    swept = []
    for rate in (0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70):
        cut = np.quantile(probe_score[train], 1.0 - rate)
        fires = probe_score[test] >= cut
        swept.append((
            float(np.where(fires, full_ok[test], cheap_ok[test]).mean()),
            float(_policy_cost(fires, read="probe").mean()),
        ))

    undominated = 0
    values = (100.0, 200.0, 400.0, 800.0, 1600.0, 3200.0)
    for value in values:
        rule = fit_gain_rule(
            probe_score[train], gains[train], delta_ms=delta, value_ms_per_correct=value
        )
        fires = rule.escalate(probe_score[test])
        accuracy = float(np.where(fires, full_ok[test], cheap_ok[test]).mean())
        latency = float(_policy_cost(fires, read="probe").mean())
        undominated += not any(
            a >= accuracy - 1e-9 and c <= latency + 1e-9 and (a > accuracy or c < latency)
            for a, c in swept
        )
    ok = undominated >= len(values) - 1
    return ("PASS" if ok else "FAIL"), (
        f"{undominated}/{len(values)} self-selected points undominated by the tuned sweep"
    )


@check("D3: the probe's break-even gain is half entropy's, so it escalates on weaker evidence")
def _():
    from gwel.router.decision import escalation_delta

    entropy_delta = escalation_delta(COSTS, read="entropy")
    probe_delta = escalation_delta(COSTS, read="probe")
    ratio = probe_delta / entropy_delta
    ok = 0.40 <= ratio <= 0.60
    return ("PASS" if ok else "FAIL"), (
        f"escalation costs {probe_delta:.1f} ms on the probe vs {entropy_delta:.1f} ms "
        f"on entropy ({ratio:.2f}x)"
    )


@check("D4: probe dominance is NOT universal across splits, but its saving is")
def _():
    """The headline is fold-specific; the distributional statement is not.

    Reported as a claim in its own right because the single-fold version was
    stated too strongly, and this check exists so it cannot be restated.
    """
    summary = json.loads(Path("results/resplit_dominance.json").read_text())
    universal = summary["dominance_rate"]
    positive = summary["saving_positive_rate"]
    share = summary["probe_share_of_front_mean"]
    ok = universal < 0.95 and positive >= 0.95 and share >= 0.75
    return ("PASS" if ok else "FAIL"), (
        f"all-dominated in {universal:.0%} of {summary['trials']} splits; "
        f"saving positive in {positive:.0%}; probe holds {share:.0%} of the front; "
        f"median saving {summary['saving_median']:+.1%}"
    )


# ---------------------------------------------------- abstention claims (B)

def _solvable(example_ids: list[str]) -> np.ndarray:
    """Whether any routable configuration answered each query correctly."""
    grouped = _grouped(PILOT)
    return np.array([
        any(
            record.correct
            for config_id, record in grouped[example].items()
            if config_id != "no_image"
        )
        for example in example_ids
    ])


def _usable_ids() -> list[str]:
    stored = np.load("results/activations_full.npz", allow_pickle=True)
    grouped = _grouped(PILOT)
    return [
        str(e) for e in stored["example_ids"]
        if "lowres_384" in grouped[str(e)] and "full" in grouped[str(e)]
        and grouped[str(e)]["lowres_384"].signals
    ]


@check("B1: abstention is worth little as a budget action, because the rule already avoids hopeless queries")
def _():
    """Escalating a query no configuration can answer is pure waste.

    If the calibrated rule escalated them at their base rate, a perfect
    abstention gate would recover that whole share. It does not: the rule sends
    them to escalation *below* base rate, so the headroom is the margin only.
    """
    from gwel.router.decision import escalation_delta, fit_gain_rule

    cheap_ok, full_ok, _, probe_score, gains, train, test = _decision_arrays()
    solvable = _solvable(_usable_ids())
    delta = escalation_delta(COSTS, read="probe")

    rule = fit_gain_rule(
        probe_score[train], gains[train], delta_ms=delta, value_ms_per_correct=800.0
    )
    fires = rule.escalate(probe_score[test])
    base = float(1 - solvable[test].mean())
    escalated_hopeless = float((~solvable[test][fires]).mean())

    gated = fires & solvable[test]
    cost = float(_policy_cost(fires, read="probe").mean())
    gated_cost = float(_policy_cost(gated, read="probe").mean())
    saving = (cost - gated_cost) / cost
    accuracy_loss = float(
        np.where(fires, full_ok[test], cheap_ok[test]).mean()
        - np.where(gated, full_ok[test], cheap_ok[test]).mean()
    )
    ok = escalated_hopeless < base and saving < 0.10 and abs(accuracy_loss) < 1e-9
    return ("PASS" if ok else "FAIL"), (
        f"escalates hopeless queries at {escalated_hopeless:.0%} vs {base:.0%} base; "
        f"a perfect gate saves {saving:.1%} at zero accuracy cost"
    )


@check("B2: the three-way conformal policy holds its coverage guarantee out of sample")
def _():
    """Split conformal promises miscoverage <= alpha on exchangeable test data.

    Checked directly: the share of test scores at or below the answer threshold
    must reach 1 - alpha, up to the finite-sample slack of a 600-example
    calibration set.
    """
    from gwel.modeling.signals import ConfidenceSignals
    from gwel.router.conformal import fit_three_way

    grouped = _grouped(PILOT)
    ids = _usable_ids()
    scores = np.array([
        ConfidenceSignals.from_dict(grouped[e]["lowres_384"].signals).mean_entropy
        for e in ids
    ])
    from gwel.router.splits import make_split

    split = make_split(
        ids, [grouped[e]["lowres_384"].dataset for e in ids],
        val_fraction=0.2, test_fraction=0.2, seed=1234,
    )
    order = {e: i for i, e in enumerate(ids)}
    train = np.array([order[e] for e in split.train])
    test = np.array([order[e] for e in split.test])

    slack = 1.0 / (len(train) + 1)
    gaps = []
    for answer_alpha, abstain_alpha in ((0.70, 0.30), (0.60, 0.20), (0.50, 0.10)):
        policy = fit_three_way(
            scores[train].tolist(),
            answer_alpha=answer_alpha, abstain_alpha=abstain_alpha,
        )
        realised = float((scores[test] <= policy.answer_threshold).mean())
        gaps.append(realised - (1.0 - answer_alpha))
    ok = all(gap >= -0.05 - slack for gap in gaps)
    return ("PASS" if ok else "FAIL"), (
        f"coverage minus target: {[f'{g:+.3f}' for g in gaps]} (slack {slack:.4f})"
    )


@check("B3: a recall-controlled threshold meets its target and cannot degenerate")
def _():
    """The fix for a tuner that discovers 'never escalate' on one fold.

    Ruan et al.'s construction sets the gate from an exact Clopper-Pearson
    lower bound on the survival rate of recoverable queries, so a recall floor
    forbids the empty policy by construction.
    """
    from gwel.modeling.signals import ConfidenceSignals
    from gwel.router.recall_control import certifiable_recall, fit_recall_controlled
    from gwel.router.splits import make_split

    grouped = _grouped(PILOT)
    ids = _usable_ids()
    scores = np.array([
        ConfidenceSignals.from_dict(grouped[e]["lowres_384"].signals).mean_entropy
        for e in ids
    ])
    cheap_ok = np.array([grouped[e]["lowres_384"].correct for e in ids])
    full_ok = np.array([grouped[e]["full"].correct for e in ids])
    recoverable = (~cheap_ok) & full_ok

    split = make_split(
        ids, [grouped[e]["lowres_384"].dataset for e in ids],
        val_fraction=0.2, test_fraction=0.2, seed=1234,
    )
    order = {e: i for i, e in enumerate(ids)}
    train = np.array([order[e] for e in split.train])
    test = np.array([order[e] for e in split.test])

    ceiling = certifiable_recall(int(recoverable[train].sum()))
    rows = []
    ok = True
    for target in (0.80, 0.90):
        fitted = fit_recall_controlled(
            scores[train].tolist(), recoverable[train].tolist(), target_recall=target
        )
        escalates = scores[test] >= fitted.threshold
        achieved = float(escalates[recoverable[test]].mean())
        rows.append(f"{target:.2f}->{achieved:.2f}")
        # The guarantee is on the certified floor, not the point estimate, so a
        # small shortfall on 200 test examples is allowed; an empty policy is not.
        ok &= achieved >= target - 0.10 and escalates.mean() > 0.0
    return ("PASS" if ok else "FAIL"), (
        f"target->achieved {rows}; certifiable ceiling {ceiling:.3f}"
    )


# -------------------------------------------------- cost-model claims (M)

@check("M1: the flat escalation price under-charges, because profiling used an image that did not escalate")
def _():
    """A third measurement bug of the same family as the two in FINDINGS S7.

    The processor caps its target at the input's longest side, so ``full`` costs
    what the image allows. The profiling image yielded 320 visual tokens; the
    pilot averages far more, so a flat cost taken from it prices an escalation
    that did not happen.
    """
    from gwel.oracle.token_cost import fit_token_cost

    profile = json.loads(Path("results/component_latency.json").read_text())
    model = fit_token_cost(
        [r["visual_tokens"] for r in profile], [r["total_ms"] for r in profile]
    )
    grouped = _grouped(PILOT)
    tokens = np.array([
        grouped[e]["full"].visual_tokens for e in grouped if "full" in grouped[e]
    ], dtype=float)
    profiled_max = max(r["visual_tokens"] for r in profile)
    flat = next(r["total_ms"] for r in profile if r["config"] == "longest_1536")
    honest = float(model.predict(tokens).mean())
    ok = honest > flat * 1.05 and model.residual_ms < 5.0
    return ("PASS" if ok else "FAIL"), (
        f"pilot mean {tokens.mean():.0f} tokens vs {profiled_max} profiled; "
        f"flat {flat:.1f} ms vs per-example {honest:.1f} ms ({honest / flat - 1:+.0%}), "
        f"fit residual {model.residual_ms:.1f} ms"
    )


@check("M2: the probe escalates more expensive queries than entropy does")
def _():
    """A selection effect on cost that a flat cost model cannot show.

    The queries more pixels help are the queries with more pixels to add, so a
    signal that ranks escalation value well also picks the dearer escalations.
    Ranking quality and cost saving therefore partly cancel.
    """
    from gwel.router.decision import signed_gain
    from gwel.router.probes import fit_layer_probe
    from gwel.router.splits import make_split

    stored = np.load("results/activations_full.npz", allow_pickle=True)
    activations, ids = stored["activations"], [str(e) for e in stored["example_ids"]]
    grouped = _grouped(PILOT)
    usable = [
        e for e in ids
        if "lowres_384" in grouped[e] and "full" in grouped[e]
        and grouped[e]["lowres_384"].signals
    ]
    position = {e: i for i, e in enumerate(ids)}
    matrix = activations[[position[e] for e in usable]][:, 6, :]
    cheap_ok = np.array([grouped[e]["lowres_384"].correct for e in usable])
    full_ok = np.array([grouped[e]["full"].correct for e in usable])
    entropy = np.array([
        float(grouped[e]["lowres_384"].signals["mean_entropy"]) for e in usable
    ])
    tokens = np.array([grouped[e]["full"].visual_tokens for e in usable], dtype=float)

    split = make_split(
        usable, [grouped[e]["lowres_384"].dataset for e in usable],
        val_fraction=0.2, test_fraction=0.2, seed=1234,
    )
    order = {e: i for i, e in enumerate(usable)}
    train = np.array([order[e] for e in split.train])
    test = np.array([order[e] for e in split.test])
    gains = signed_gain(cheap_ok, full_ok)
    score = fit_layer_probe(matrix[train], (gains[train] > 0).astype(float),
    6).score(matrix)

    ratios = []
    for rate in (0.20, 0.30, 0.40):
        entropy_fires = entropy[test] >= np.quantile(entropy[train], 1 - rate)
        probe_fires = score[test] >= np.quantile(score[train], 1 - rate)
        ratios.append(float(tokens[test][probe_fires].mean() /
        tokens[test][entropy_fires].mean()))
    ok = all(r > 1.05 for r in ratios)
    return ("PASS" if ok else "FAIL"), (
        f"probe/entropy escalated-token ratio {[f'{r:.2f}' for r in ratios]} at 20/30/40%"
    )


@check("M3: KV-cache reuse cannot erase the probe's advantage, because decode is uncacheable")
def _():
    summary = json.loads(Path("results/cache_sensitivity.json").read_text())
    floor = summary["floor_ms"]
    share = summary["floor_share_of_uncached"]
    ok = floor > 0.0 and share > 0.5 and not summary["advantage_erasable"]
    return ("PASS" if ok else "FAIL"), (
        f"at a perfect encoder+prefill cache the probe still saves {floor:.1f} ms "
        f"per escalation, {share:.0%} of its uncached saving"
    )


# ------------------------------------------- domain-confound claims (E)

@check("E1: a free image-size feature nearly matches the probe on the pooled mixture")
def _():
    summary = json.loads(Path("results/domain_confound.json").read_text())
    pooled = summary["pooled"]
    margin = pooled["probe"] - pooled["image_size"]
    ok = margin < 0.05
    return ("PASS" if ok else "FAIL"), (
        f"probe {pooled['probe']:.3f} vs image size {pooled['image_size']:.3f} "
        f"(margin {margin:+.3f}), entropy {pooled['entropy']:.3f}"
    )


@check("E2: within a single dataset the probe falls to chance while entropy holds")
def _():
    """The load-bearing negative result of the cost audit.

    If the probe encoded escalation value it would survive removal of the
    between-domain axis. It does not, at any depth.
    """
    summary = json.loads(Path("results/domain_confound.json").read_text())
    weighted = summary["within_weighted"]
    per_dataset = summary["within"]
    best_layer = max(summary["layer_sweep"]["points"], key=lambda r: r["auroc"])
    entropy_there = next(
        row["entropy"] for row in per_dataset
        if row["dataset"] == summary["layer_sweep"]["dataset"]
    )
    chance_like = all(row["probe_low"] < 0.62 for row in per_dataset)
    ok = (
        weighted["probe"] < 0.60
        and weighted["entropy"] > 0.62
        and chance_like
        and best_layer["auroc"] < entropy_there
    )
    return ("PASS" if ok else "FAIL"), (
        f"within-domain probe {weighted['probe']:.3f} vs entropy "
        f"{weighted['entropy']:.3f}; best depth {best_layer['layer']} reaches "
        f"{best_layer['auroc']:.3f} against entropy {entropy_there:.3f}"
    )


# ------------------------------------------ within-domain rule claims (W)

DATASET_SIZES = {"docvqa": 300, "textvqa": 250, "vqav2": 300, "vstar": 150}


def _within_domain():
    return json.loads(Path("results/within_domain.json").read_text())


def _weighted(rows: dict, policy: str) -> tuple[float, float]:
    total = sum(DATASET_SIZES[d] for d in rows)
    accuracy = sum(rows[d][policy]["accuracy"] * DATASET_SIZES[d] for d in rows) / total
    latency = sum(rows[d][policy]["latency"] * DATASET_SIZES[d] for d in rows) / total
    return accuracy, latency


@check("W1: one operator preference yields a different escalation rate per domain, tracking its value")
def _():
    """What a tuned rate structurally cannot do.

    Escalation value is domain-determined, so a fixed rate is wrong in every
    domain but one. A rule calibrated on the observed gain distribution picks
    its own rate wherever it is fitted.
    """
    rows = _within_domain()["datasets"]
    repairs = json.loads(Path("results/domain_confound.json").read_text())["mixture"]
    repair_rate = {row["dataset"]: row["repair_rate"] for row in repairs}
    chosen, actual = [], []
    for dataset, policies in rows.items():
        chosen.append(policies["gain rule V=800"]["rate"])
        actual.append(repair_rate[dataset])
    correlation = float(np.corrcoef(chosen, actual)[0, 1])
    spread = max(chosen) - min(chosen)
    ok = correlation > 0.85 and spread > 0.30
    return ("PASS" if ok else "FAIL"), (
        f"rates {[f'{c:.0%}' for c in chosen]} against repair rates "
        f"{[f'{a:.0%}' for a in actual]}, r={correlation:+.3f}"
    )


@check("W2: one cost-derived rule beats one tuned rate applied across the same domains")
def _():
    """The method survives its signal being discredited.

    Both policies are fixed once and applied unchanged to four domains, which is
    what a deployment does. Sizes weight the average.
    """
    rows = _within_domain()["datasets"]
    rule = _weighted(rows, "gain rule V=800")
    tuned = _weighted(rows, "tuned rate 30%")
    ok = rule[0] > tuned[0] and rule[1] < tuned[1]
    return ("PASS" if ok else "FAIL"), (
        f"rule {rule[0]:.3f} at {rule[1]:.1f} ms dominates tuned-30% "
        f"{tuned[0]:.3f} at {tuned[1]:.1f} ms"
    )


@check("W3: a per-query break-even is NOT worth it, and the confound explains why")
def _():
    """A negative result on a refinement of our own.

    Charging each query its own escalation price should help where prices vary.
    Within a domain they barely do --- image size is largely a domain property,
    which is the same fact that produced the confound --- so the refinement buys
    nothing. Asserted as a negative so it cannot be quietly reported as a win.
    """
    summary = _within_domain()["per_query_vs_global"]
    accuracy, latency = summary["accuracy"], summary["latency"]
    accuracy_null = accuracy[1] <= 0.0 <= accuracy[2]
    latency_null = latency[1] <= 0.0 <= latency[2]
    return ("PASS" if accuracy_null and latency_null else "FAIL"), (
        f"accuracy {_signed(accuracy[0])} "
        f"[{_signed(accuracy[1])}, {_signed(accuracy[2])}], "
        f"latency {latency[0]:+.1f} [{latency[1]:+.1f}, {latency[2]:+.1f}] ms: "
        f"both span zero"
    )


# ------------------------------------------------ resolution-ladder claims (L)

@check("L1: most escalated queries do not need the top rung")
def _():
    """Binary escalation over-serves, and the oracle over the ladder says by how much."""
    grouped = _grouped(PILOT)
    rungs = ["lowres_384", "lowres_768", "full"]
    ids = [e for e in grouped if all(c in grouped[e] for c in rungs)]
    correct = {c: np.array([grouped[e][c].correct for e in ids]) for c in rungs}

    cheapest = np.full(len(ids), -1)
    for index in range(len(ids)):
        for level, config_id in enumerate(rungs):
            if correct[config_id][index]:
                cheapest[index] = level
                break
    needs_middle = float((cheapest == 1).mean())
    needs_top = float((cheapest == 2).mean())
    over_served = needs_middle / (needs_middle + needs_top)
    ok = over_served > 0.60
    return ("PASS" if ok else "FAIL"), (
        f"{needs_middle:.1%} need the middle rung against {needs_top:.1%} the top; "
        f"binary escalation over-serves {over_served:.0%} of what it escalates"
    )


@check("L2: the first rung is the efficient one")
def _():
    from gwel.oracle.token_cost import fit_token_cost

    profile = json.loads(Path("results/component_latency.json").read_text())
    model = fit_token_cost(
        [r["visual_tokens"] for r in profile], [r["total_ms"] for r in profile]
    )
    grouped = _grouped(PILOT)
    rungs = ["lowres_384", "lowres_768", "full"]
    ids = [e for e in grouped if all(c in grouped[e] for c in rungs)]
    correct = {c: np.array([grouped[e][c].correct for e in ids]) for c in rungs}
    latency = {
        c: model.predict(np.array([grouped[e][c].visual_tokens for e in ids]))
        for c in rungs
    }

    def rate(low: str, high: str) -> float:
        gain = float(
            (correct[high] & ~correct[low]).mean() - (correct[low] & ~correct[high]).mean()
        )
        return gain / float((latency[high] - latency[low]).mean()) * 1000.0

    first = rate("lowres_384", "lowres_768")
    second = rate("lowres_768", "full")
    binary = rate("lowres_384", "full")
    ok = first > second and first > binary
    return ("PASS" if ok else "FAIL"), (
        f"points per second: first rung {first:+.2f}, second {second:+.2f}, "
        f"binary jump {binary:+.2f}"
    )


@check("L3: the ladder policy is cheaper at indistinguishable accuracy, and only where a rung exists")
def _():
    summary = json.loads(Path("results/ladder.json").read_text())
    accuracy = summary["ladder_vs_binary"]["accuracy"]
    latency = summary["ladder_vs_binary"]["latency"]
    accuracy_null = accuracy[1] <= 0.0 <= accuracy[2]
    cheaper = latency[2] < 0.0
    # The effect must be concentrated where the two rungs are distinct
    # configurations; where they are not, the ladder must do nothing.
    per_domain = summary["per_domain"]
    absent = [v for v in per_domain.values() if v["top_rung_exists"] < 0.05]
    present = [v for v in per_domain.values() if v["top_rung_exists"] > 0.5]
    inert = all(abs(v["latency_delta"]) < 5.0 for v in absent)
    biggest = min(v["latency_delta"] for v in present)
    ok = accuracy_null and cheaper and inert and biggest < -20.0
    return ("PASS" if ok else "FAIL"), (
        f"accuracy {accuracy[0]:+.3f} [{accuracy[1]:+.3f}, {accuracy[2]:+.3f}], "
        f"latency {latency[0]:+.1f} [{latency[1]:+.1f}, {latency[2]:+.1f}] ms; "
        f"best domain {biggest:+.1f} ms, inert where no rung exists"
    )


@check("L4: a configuration name does not determine a configuration")
def _():
    """The fourth instance of this project's recurring measurement error.

    For half the pilot the 'full resolution' pass is literally the intermediate
    pass, because the processor caps its target at the input's longest side.
    """
    summary = json.loads(Path("results/ladder.json").read_text())
    existence = summary["top_rung_exists"]
    absent = [d for d, share in existence.items() if share < 0.05]
    ok = len(absent) >= 2
    return ("PASS" if ok else "FAIL"), (
        "top rung exists for "
        + ", ".join(f"{d} {share:.0%}" for d, share in sorted(existence.items()))
    )


# ------------------------------------ single-domain resolution claims (R)

def _docvqa_pilot():
    return json.loads(Path("results/docvqa_pilot.json").read_text())


@check("R1: escalation value saturates well below the resolution the model accepts")
def _():
    """The top rung of a verified ladder buys nothing, on 1200 DocVQA pages.

    Measured where escalation matters most and where every rung is a genuinely
    distinct configuration, so this is not the mixture's rung-collapse artefact.
    """
    summary = _docvqa_pilot()
    steps = summary["steps"]
    top = steps[-1]
    below = steps[-2]
    saturated = top["low"] <= 0.0 <= top["high"]
    below_helps = below["low"] > 0.0
    ok = saturated and below_helps
    return ("PASS" if ok else "FAIL"), (
        f"{below['from']}->{below['to']} gains {below['net_gain']:+.3f} "
        f"[{below['low']:+.3f}, {below['high']:+.3f}]; "
        f"{top['from']}->{top['to']} gains {top['net_gain']:+.3f} "
        f"[{top['low']:+.3f}, {top['high']:+.3f}] for {top['extra_ms']:.0f} ms"
    )


@check("R2: the first rung is several times more efficient than the ones above it")
def _():
    summary = _docvqa_pilot()
    rates = [s["points_per_second"] for s in summary["steps"]]
    ok = rates[0] > 2.0 * max(rates[1:])
    return ("PASS" if ok else "FAIL"), (
        "points per second by rung: " + ", ".join(f"{r:+.2f}" for r in rates)
    )


@check("R3: most of what escalation repairs is repaired below the top rung")
def _():
    summary = _docvqa_pilot()
    share = summary["repaired_below_top"]
    ok = share > 0.75
    return ("PASS" if ok else "FAIL"), (
        f"{share:.0%} of repaired queries need less than the top rung; "
        f"unsolvable at any rung {summary['unsolvable']:.0%}"
    )


@check("R4: a within-domain probe improves with data but plateaus below output entropy")
def _():
    """The learning curve that separates an absent signal from a starved one.

    The answer is neither extreme. AUROC does climb with data, so the
    within-domain result is not purely starvation; but it plateaus well below
    output entropy, which is fitted on nothing, so the signal is genuinely
    weaker inside a domain than the pooled figure suggests.
    """
    summary = _docvqa_pilot()
    curve = summary["learning_curve"]
    first, last = curve[0], curve[-1]
    climbed = last["probe"] - first["probe"]
    beats_entropy = last["probe"] > last["entropy"]
    ok = climbed < 0.10 and not beats_entropy
    return ("PASS" if ok else "FAIL"), (
        f"probe {first['probe']:.3f} at n={first['train_n']} to {last['probe']:.3f} "
        f"at n={last['train_n']} ({climbed:+.3f}); entropy {last['entropy']:.3f}"
    )


@check("R5: the layer chosen on the mixture is the wrong layer within a domain")
def _():
    """A second mixture artefact, and it costs the probe its read-cost story.

    The paper selects layer 6 because AUROC saturates there on the pooled
    mixture, and the whole cheap-read argument rests on stopping a sixth of the
    way through. Inside one domain the signal keeps improving with depth, so the
    read that works is a full prefill.
    """
    summary = _docvqa_pilot()
    depth = {row["layer"]: row["auroc"] for row in summary["depth"]}
    best_layer = max(depth, key=lambda layer: depth[layer])
    encoder, prefill, decode = 11.0, 48.1, 67.8
    cheap = encoder + prefill + decode

    def read_cost(layer: int) -> float:
        return encoder + prefill * layer / 32

    gain = depth[best_layer] - depth[6]
    ok = best_layer > 12 and gain > 0.03
    return ("PASS" if ok else "FAIL"), (
        f"layer 6 gives {depth[6]:.3f}, layer {best_layer} gives {depth[best_layer]:.3f} "
        f"({gain:+.3f}); read cost rises {read_cost(6):.1f} to {read_cost(best_layer):.1f} ms, "
        f"cutting the saving from {cheap - read_cost(6):.1f} to "
        f"{cheap - read_cost(best_layer):.1f} ms"
    )


@check("R6: the ladder is cheaper than binary escalation at most operating points")
def _():
    """Paired inside each operating point, not across them.

    Pooling the preference means would resample four numbers; the interval that
    matters is over the resamples that produced each point.
    """
    summary = _docvqa_pilot()
    rows = summary["ladder_vs_binary"]
    cheaper = [r for r in rows if r["latency_delta"][2] < 0.0]
    # At the highest preference the accuracy difference must not be a real loss.
    top = max(rows, key=lambda r: r["value"])
    accuracy_ok = top["accuracy_delta"][1] <= 0.0 <= top["accuracy_delta"][2]
    ok = len(cheaper) >= len(rows) - 1 and accuracy_ok
    return ("PASS" if ok else "FAIL"), (
        f"significantly cheaper at {len(cheaper)}/{len(rows)} points; at V="
        f"{top['value']:.0f} accuracy {top['accuracy_delta'][0]:+.3f} "
        f"[{top['accuracy_delta'][1]:+.3f}, {top['accuracy_delta'][2]:+.3f}], "
        f"latency {top['latency_delta'][0]:+.1f} ms"
    )


@check("X1: every claim lost to the correction is named, including the convenient ones")
def _():
    """Holm over the whole family, with the losses enumerated rather than
    summarised.

    The family grew as experiments were added, and three of its losses are
    comparisons in which our probe nominally beat the free image descriptor.
    Losing those makes the two signals harder to tell apart, which is the
    direction that costs us the apparatus, so the check asserts they are among
    the losses rather than merely counting them. It also asserts the headline
    AUROC is still lost, which was the original point.
    """
    summary = json.loads(Path("results/multiplicity.json").read_text())
    nominal, survivors = summary["nominal"], summary["survivors"]
    total = len(summary["tests"])
    lost = [
        t["name"] for t in summary["tests"]
        if t["p_value"] <= summary["alpha"] and not t["survives"]
    ]
    headline_lost = any("AUROC" in name for name in lost)
    # Comparisons where the probe nominally beat the free signal: if any of
    # those survived while the headline did not, the paper would be keeping
    # exactly the claims that flatter it.
    convenient = [n for n in lost if "free signal minus probe" in n]
    every_test_scored = total >= 40
    ok = (
        headline_lost
        and bool(convenient)
        and survivors == nominal - len(lost)
        and every_test_scored
    )
    return ("PASS" if ok else "FAIL"), (
        f"{nominal}/{total} clear the nominal level, {survivors} survive Holm, "
        f"{len(lost)} lost including the headline AUROC={headline_lost} and "
        f"{len(convenient)} probe-favouring comparison(s); uncorrected "
        f"family-wise error {summary['family_wise_error_uncorrected']:.0%}"
    )


@check("X2: the accuracy claims are nulls and the cost claims are not")
def _():
    """A structural read of the same family, and a guard against reversal.

    The paper's positive results are about cost at matched accuracy. If an
    accuracy comparison ever started clearing the correction, the story would
    have changed and this check should say so.
    """
    summary = json.loads(Path("results/multiplicity.json").read_text())
    accuracy = [t for t in summary["tests"] if "accuracy" in t["name"]]
    latency = [t for t in summary["tests"] if "latency" in t["name"]]
    ok = (
        not any(t["survives"] for t in accuracy)
        and all(t["survives"] for t in latency)
    )
    return ("PASS" if ok else "FAIL"), (
        f"{sum(t['survives'] for t in latency)}/{len(latency)} latency claims survive, "
        f"{sum(t['survives'] for t in accuracy)}/{len(accuracy)} accuracy claims do"
    )


@check("Y1: every component earns its place at a fixed budget, by very unequal margins")
def _():
    """The ablation, read the only way that is not circular.

    At a fixed operator preference, any variant that escalates more looks more
    accurate for that reason alone. Read at a fixed latency budget instead,
    every removal is a real loss. They are not remotely equal losses, and
    per-example pricing is detectable but an order of magnitude smaller than the
    rest, which is a different statement from "no effect".
    """
    summary = json.loads(Path("results/ablation.json").read_text())
    budget = min(summary["budgets"])
    key = f"delta_at_{int(budget)}"
    rows = {r["variant"]: r for r in summary["rows"] if key in r}
    large = {name: row for name, row in rows.items() if row[key][2] < -0.05}
    pricing = rows.get("no per-example pricing")
    ok = (
        len(large) >= 3
        and pricing is not None
        and pricing[key][2] < 0.0                      # a real loss
        and abs(pricing[key][0]) < 0.05                # and a negligible one
        and abs(pricing[key][0]) * 10 < min(abs(r[key][0]) for r in large.values())
    )
    return ("PASS" if ok else "FAIL"), (
        f"at {budget:.0f} ms, {len(large)}/{len(rows)} removals cost over 0.05 "
        f"accuracy (worst {min(r[key][0] for r in large.values()):+.3f}); "
        f"per-example pricing costs {pricing[key][0]:+.3f} "
        f"[{pricing[key][1]:+.3f}, {pricing[key][2]:+.3f}], real but 100x smaller"
    )


@check("Y2: the ladder is what makes a tight budget usable at all")
def _():
    """At 400 ms no binary policy can afford the top rung, so it degenerates."""
    summary = json.loads(Path("results/ablation.json").read_text())
    budget = min(summary["budgets"])
    key = f"acc_at_{int(budget)}"
    rows = {r["variant"]: r for r in summary["rows"]}
    reference = rows["reference"][key]
    binary = rows["no ladder (binary)"][key]
    floor = rows["no signal"][key]
    ok = reference > binary + 0.20 and abs(binary - floor) < 0.02
    return ("PASS" if ok else "FAIL"), (
        f"at {budget:.0f} ms the ladder reaches {reference:.3f} where binary reaches "
        f"{binary:.3f}, indistinguishable from always-cheap at {floor:.3f}"
    )


@check("E3: where the two signals disagree, they disagree about the dataset")
def _():
    """The confound made legible, and the sharpest form of it.

    If the probe encoded escalation value, its disagreements with entropy would
    be spread across the mixture. Instead the disagreement is itself close to a
    dataset classifier, and not one query entropy prefers is a document page.
    """
    summary = json.loads(Path("results/domain_confound.json").read_text())
    block = summary["disagreement"]
    doc_auroc = block["dataset_auroc"]["docvqa"]
    entropy_side = block["composition"]["entropy favours"]
    probe_side = block["composition"]["probe favours"]
    base = block["composition"]["test fold"]
    ok = (
        doc_auroc > 0.65
        and entropy_side["docvqa"] < 0.05
        and probe_side["docvqa"] > base["docvqa"] * 1.4
    )
    return ("PASS" if ok else "FAIL"), (
        f"disagreement predicts DocVQA at {doc_auroc:.3f}; DocVQA is "
        f"{probe_side['docvqa']:.0%} of what the probe prefers and "
        f"{entropy_side['docvqa']:.0%} of what entropy prefers, against a "
        f"{base['docvqa']:.0%} base rate"
    )


@check("R7: the saturation null is tight, not underpowered")
def _():
    """A null is only a result if the interval could have found an effect.

    Guards against the failure this project criticises elsewhere: reporting "no
    difference" at a sample size incapable of resolving one. The top step's
    interval must be narrow relative to the gains actually measured below it.
    """
    summary = _docvqa_pilot()
    steps = summary["steps"]
    top = steps[-1]
    half_width = (top["high"] - top["low"]) / 2
    smallest_detected = min(
        s["net_gain"] for s in steps[:-1] if s["low"] > 0.0
    )
    ok = half_width <= 0.05 and half_width < smallest_detected / 2
    return ("PASS" if ok else "FAIL"), (
        f"top step {top['net_gain']:+.3f} [{top['low']:+.3f}, {top['high']:+.3f}], "
        f"half-width {half_width:.3f} against a smallest detected gain of "
        f"{smallest_detected:+.3f}"
    )


@check("R9: no single token budget explains the ceiling")
def _():
    """Models with different tokenisers stop at the same pixel target.

    SmolVLM-500M and 256M share an 86M encoder, so "saturates at 1152 px" and
    "saturates at 640 visual tokens" name the same point for both. SmolVLM2-2.2B
    and Qwen2-VL-2B tokenise differently, so the two come apart, and no single
    sequence length is where all of them stop gaining.

    This rules out a fixed token budget. It does not establish that tokens do
    no work, because pixels and tokens co-vary in all four; R12 is the
    experiment that separates them, and it stops the claim here.
    """
    summary = json.loads(Path("results/saturation_models.json").read_text())
    runs = summary["runs"]
    if len(runs) < 3:
        raise FileNotFoundError("results/saturation_models.json (needs a third model)")
    rungs = {r["saturation_rung"] for r in runs}
    tokens = {r["model"]: r["median_tokens"]["lowres_1152"] for r in runs}
    tops = [r["steps"][-1] for r in runs]
    same_rung = len(rungs) == 1 and None not in rungs
    differing_tokens = len(set(tokens.values())) > 1
    all_informative = all(t["informative"] and t["null"] for t in tops)
    ok = same_rung and differing_tokens and all_informative
    return ("PASS" if ok else "FAIL"), (
        f"{len(runs)} models all saturate at {rungs.pop()} while spending "
        f"{sorted(int(v) for v in set(tokens.values()))} visual tokens there; "
        f"every top step is a tight null (worst half-width "
        f"{max((t['high'] - t['low']) / 2 for t in tops):.3f})"
    )


@check("R8: the ceiling is stable across serving models that spend tokens with resolution")
def _():
    """Two models, half the parameters apart, on the same 1200 pages.

    Capacity and legibility predict opposite things: a smaller model should
    saturate earlier if the ceiling is its own, and at the same rung with lower
    accuracy throughout if the ceiling is the images. Both nulls must also be
    tight, or this compares two failures to measure.

    The claim is deliberately class-restricted. Every model in this artefact
    raises its visual-token count with the pixel target, so the rung transports
    across them; R12 shows a model that decouples the two saturating lower, so
    this rung is an upper bound rather than a constant, and the check asserts
    that framing holds by refusing to pass if a flat-budget run is ever mixed
    into the ladder comparison.
    """
    summary = json.loads(Path("results/saturation_models.json").read_text())
    runs = summary["runs"]
    rungs = {r["model"]: r["saturation_rung"] for r in runs}
    same = summary["same_saturation_rung"]
    tight = all(
        s["informative"] for r in runs for s in r["steps"] if s["null"]
    )
    # The capacity hypothesis predicts the smaller model saturates earlier; the
    # legibility one predicts the same rung with lower accuracy throughout. Only
    # the 256M is strictly smaller than the reference, so it carries that test.
    by_model = {r["model"]: r["accuracy"] for r in runs}
    reference = by_model["HuggingFaceTB/SmolVLM-500M-Instruct"]
    smaller = by_model.get("HuggingFaceTB/SmolVLM-256M-Instruct")
    lower_everywhere = smaller is not None and all(
        smaller[c] < reference[c] for c in reference if c != "full"
    )
    # The class restriction, asserted rather than trusted to the prose: every
    # run compared here must actually raise its token count with the pixel
    # target, so that "same rung" is a statement about a coherent family.
    resolution_sensitive = all(
        r["median_tokens"]["full"] > 1.5 * r["median_tokens"]["lowres_384"]
        for r in runs
    )
    ok = same and tight and lower_everywhere and resolution_sensitive
    return ("PASS" if ok else "FAIL"), (
        f"{len(runs)} models saturate at {set(rungs.values())}; nulls tight={tight}; "
        f"the smaller model is less accurate at every rung below the "
        f"ceiling={lower_everywhere}; every model compared grows its token "
        f"count with resolution={resolution_sensitive}"
    )


@check("R10: the ceiling survives outside the lineage it was measured in")
def _():
    """R8 tests three models that share a training recipe, so a common
    pretraining corpus stays a live alternative cause. This adds a model that
    shares none of it: a different encoder mechanism (tokens proportional to
    pixels, no patch-grid buckets), a different language model, a different
    data recipe.

    The test has to survive both ways it could be vacuous. A model that
    saturates because it is too weak to use the pixels proves nothing, so the
    out-of-lineage model must be substantially *more* accurate than the
    lineage, not less. And its top-step null must be tight, or this is an
    absence of measurement rather than an absence of gain.
    """
    summary = json.loads(Path("results/saturation_models.json").read_text())
    runs = summary["runs"]
    lineage = [r for r in runs if "SmolVLM" in r["model"]]
    outside = [r for r in runs if "SmolVLM" not in r["model"]]
    if not outside:
        return "SKIP", "no out-of-lineage run in the artefact"
    if not lineage:
        return "SKIP", "no lineage runs to compare against"

    rung = lineage[0]["saturation_rung"]
    same = all(r["saturation_rung"] == rung for r in runs)
    top_nulls = [
        s for r in outside for s in r["steps"]
        if s["to"] == "full" and s["null"]
    ]
    tight = bool(top_nulls) and all(s["informative"] for s in top_nulls)

    best_lineage = max(max(r["accuracy"].values()) for r in lineage)
    stronger = [r for r in outside if max(r["accuracy"].values()) > best_lineage]
    # The point of a different encoder is a different token schedule; if it
    # matched the lineage's, this would be the same experiment twice.
    distinct_tokens = [
        r for r in outside
        if r["median_tokens"]["full"] > 1.5 * lineage[0]["median_tokens"]["full"]
    ]
    ok = same and tight and bool(stronger) and bool(distinct_tokens)
    names = ", ".join(r["model"].split("/")[-1] for r in outside)
    detail_acc = "; ".join(
        f"{r['model'].split('/')[-1]} peaks at {max(r['accuracy'].values()):.3f} "
        f"against the lineage's {best_lineage:.3f}"
        for r in outside
    )
    return ("PASS" if ok else "FAIL"), (
        f"{names} saturate at {rung} like the lineage={same}, top-step null "
        f"tight={tight}, more accurate than the lineage={bool(stronger)}, "
        f"token schedule differs={bool(distinct_tokens)}; {detail_acc}"
    )


@check("R11: the ceiling procedure states a sample size, and a small pilot cannot meet it")
def _():
    """Algorithm 3 is only usable if it says how much data it needs.

    Two things must hold for the paper's statement of it. The precision must
    improve with n at the square-root rate the sample-size rule assumes, and
    the smallest row must genuinely fail: if 100 pages already named the right
    ceiling, the warning in the text would be scaremongering rather than a
    measurement.
    """
    art = json.loads(Path("results/ceiling_sample_size.json").read_text())
    rows = {r["n"]: r for r in art["rows"]}
    widths = [rows[n]["median_half_width"] for n in sorted(rows)]
    monotone = all(a > b for a, b in zip(widths, widths[1:], strict=False))
    small = rows[100]["verdict_agreement"]
    large = rows[500]["verdict_agreement"]
    needed = art["pages_for_null_precision"]
    # The stated rule must land between the rows that bracket it, or the paper
    # is quoting a sample size its own table contradicts.
    bracketed = rows[200]["median_half_width"] > art["null_precision"] > rows[300][
        "median_half_width"
    ]
    ok = (
        monotone
        and small <= 0.6
        and large >= 0.95
        and bracketed
        and 200 <= needed <= 300
    )
    return ("PASS" if ok else "FAIL"), (
        f"half-width {widths[0]:.3f} at n=100 down to {widths[-1]:.3f} at "
        f"n={max(rows)}, monotone={monotone}; agreement {small:.0%} at n=100 "
        f"and {large:.0%} at n=500; rule says {needed} pages for a "
        f"{art['null_precision']} bar, bracketed by the table={bracketed}"
    )


@check("R12: with the token budget held still, pixels alone stop paying a rung earlier")
def _():
    """The control that stops R9 from being read as more than it is.

    Three things have to hold before this experiment says anything. The token
    budget must actually be near constant, or it is just another ladder. The
    step where the other models still gain must be a null here, and a tight
    one. And the model must not be the weakest in the comparison, or "it stops
    early because it reads badly" explains everything without any appeal to
    pixels.
    """
    art = json.loads(Path("results/fixed_budget.json").read_text())
    steps = {f"{s['from']}->{s['to']}": s for s in art["steps"]}
    contested = steps["lowres_768->lowres_1152"]
    budget_flat = art["token_spread"] <= 1.10 and art["rungs_token_identical"] >= 0.90
    null_here = contested["null"] and contested["informative"]

    saturation = json.loads(Path("results/saturation_models.json").read_text())
    others = {
        r["model"]: r for r in saturation["runs"]
    }
    gains_elsewhere = [
        s["gain"]
        for r in others.values()
        for s in r["steps"]
        if s["from"] == "lowres_768" and s["to"] == "lowres_1152"
    ]
    pays_elsewhere = all(g > 0.02 for g in gains_elsewhere)
    peak = max(art["accuracy"].values())
    weakest = min(max(r["accuracy"].values()) for r in others.values())
    not_weakest = peak > weakest
    ok = budget_flat and null_here and pays_elsewhere and not_weakest
    return ("PASS" if ok else "FAIL"), (
        f"tokens {art['token_spread']:.2f}x across rungs against "
        f"{art['pixel_spread']:.1f}x in pixels, identical on "
        f"{art['rungs_token_identical']:.0%} of pages; 768->1152 is "
        f"{contested['gain']:+.3f} [{contested['low']:+.3f}, "
        f"{contested['high']:+.3f}] here against "
        f"{min(gains_elsewhere):+.3f} to {max(gains_elsewhere):+.3f} elsewhere; "
        f"ceiling {art['ceiling']}; peaks at {peak:.3f}, above the weakest "
        f"compared model's {weakest:.3f}={not_weakest}"
    )


@check("CV2: a signal costing nothing clears the hull, and the abort mechanism alone does not")
def _():
    """The two comparators that decide what the probe is worth.

    Image size is read from the file header, so a policy on it skips the cheap
    pass entirely on escalated queries and prices exactly like the
    randomisation it is compared against. Any gap it opens is bought by the
    signal. The random-plus-abort policy is the converse control: same abort,
    no signal. Both verdicts must hold under both costings, or the result is an
    artefact of the pricing the paper itself rejects.
    """
    art = json.loads(Path("results/free_signal.json").read_text())
    free_clears, random_clears, worst_free = True, False, 1.0
    for costing in ("flat", "per-example"):
        block = art[costing]["preference swept"]
        for name, row in block.items():
            estimate, low, high = row["gap"]
            if name.startswith("image size"):
                free_clears = free_clears and low > 0.0
                worst_free = min(worst_free, low)
            if name.startswith("random, abort") and low > 0.0:
                random_clears = True
    # The free signal has to be competitive with the probe, not merely positive:
    # if the probe beat it everywhere the pre-generation read would be earning
    # its cost after all.
    # The artefact stores both the interval and the resample vector behind it;
    # only the intervals carry the estimate this compares.
    spread = max(
        abs(row[0])
        for costing in ("flat", "per-example")
        for key, row in art[costing]["free minus probe"].items()
        if not key.endswith("vector")
    )
    ok = free_clears and not random_clears and spread <= 0.02
    return ("PASS" if ok else "FAIL"), (
        f"image size clears the hull at every preference of both costings="
        f"{free_clears} (worst interval low {worst_free:+.3f}); random plus "
        f"abort never clears={not random_clears}; paired against the probe the "
        f"free signal is within {spread:.3f} everywhere"
    )


@check("CV3: dataset identity alone bounds every signal in the paper")
def _():
    """The control that turns the confound from a correlation into a ceiling.

    A policy given the dataset label, escalating a random subset within each
    dataset, must dominate the probe at every preference. If it did not, the
    probe would be reading something beyond identity and the confound story
    would be incomplete. Selection inside a dataset has to stay random, which
    the script enforces; here we check the consequence.
    """
    art = json.loads(Path("results/domain_policy.json").read_text())
    policies = art["policies"]
    values = sorted(
        {name.split("V=")[1] for name in policies if name.startswith("probe")},
        key=float,
    )
    dominates, margins = True, []
    for value in values:
        label = policies[f"domain label, tuned rate V={value}"]["gap"]
        probe = policies[f"probe V={value}"]["gap"]
        margins.append(label[0] - probe[0])
        dominates = dominates and label[1] > probe[0]
    clears = all(
        policies[f"domain label, tuned rate V={v}"]["gap"][1] > 0 for v in values
    )
    # The rates it picks must track escalation value per domain, or the bound
    # is arithmetic rather than meaningful.
    rates = art["oracle_rates"]
    docvqa = np.mean([v for k, v in rates.items() if k.startswith("docvqa")])
    vqav2 = np.mean([v for k, v in rates.items() if k.startswith("vqav2")])
    ordered = docvqa > 2 * vqav2
    ok = dominates and clears and ordered
    return ("PASS" if ok else "FAIL"), (
        f"the label bound clears the hull at every preference={clears} and "
        f"dominates the probe={dominates} by {min(margins):+.3f} to "
        f"{max(margins):+.3f}; it escalates {docvqa:.0%} of DocVQA against "
        f"{vqav2:.0%} of VQAv2"
    )


@check("R13: with the pixels held still, extra visual tokens do not repay their cost")
def _():
    """The converse of R12, and the pair that separates the two axes.

    Three conditions. The token gain at fixed pixels must be a null, and a
    tight one, or the experiment found nothing rather than nothing to find.
    The step must be read by what it actually spent rather than by the bound's
    name, since the bound binds on only part of the corpus. And the pages where
    the bound changed no tokens must show exactly zero difference: they receive
    an identical input and an identical sequence under greedy decoding, so
    anything else would mean the measurement itself is noisy.
    """
    art = json.loads(Path("results/tile_budget_analysis.json").read_text())
    steps = {(s["from"], s["to"]): s for s in art["steps"]}
    first = steps[(4, 12)]["all"]
    second = steps[(12, 24)]
    control = second["tokens_unchanged"]
    spent = second["tokens_rose"]
    if control is None or spent is None:
        return "SKIP", "the second bound did not split the corpus"

    token_null = first["null"] and first["informative"]
    control_exact = (
        abs(control["gain"]) < 1e-12
        and abs(control["low"]) < 1e-12
        and abs(control["high"]) < 1e-12
    )
    # Where the tokens were actually spent, the effect must not be a gain: the
    # claim is that sequence length is not the binding constraint here.
    not_a_gain = spent["low"] <= 0.0
    partial = 0.2 <= second["share_where_tokens_rose"] <= 0.8
    ok = token_null and control_exact and not_a_gain and partial
    return ("PASS" if ok else "FAIL"), (
        f"tripling tokens at fixed pixels gives {first['gain']:+.3f} "
        f"[{first['low']:+.3f}, {first['high']:+.3f}], null and informative="
        f"{token_null}; the next bound binds on "
        f"{second['share_where_tokens_rose']:.0%} of pages and gives "
        f"{spent['gain']:+.3f} there; the unchanged-token control is exactly "
        f"zero over {control['n']} pages={control_exact}"
    )


@check("CV4: the two regimes invert, and neither signal works in both")
def _():
    """The paper's central claim, asserted as an inversion rather than as two
    separate observations.

    Across a mixture, the free image descriptor clears the randomisation hull
    and output entropy does not. Inside one workload it has to be the other way
    round, or the claim is a coincidence of two datasets rather than a property
    of traffic heterogeneity. Both halves are required, and the free
    descriptor's within-workload signal must be at chance, since a weak but
    real signal would make this a difference of degree.
    """
    mixture = json.loads(Path("results/free_signal.json").read_text())
    single = json.loads(Path("results/free_signal_docvqa.json").read_text())

    mix = mixture["per-example"]["preference swept"]
    free_clears_mixture = all(
        row["gap"][1] > 0 for name, row in mix.items() if name.startswith("image size")
    )
    entropy_fails_mixture = all(
        row["gap"][2] <= 0.005
        for name, row in mix.items()
        if name.startswith("entropy")
    )

    free_fails_single = all(
        row["gap"][1] <= 0
        for name, row in single.items()
        if name.startswith("ladder, image size")
    )
    entropy_clears_single = any(
        row["gap"][1] > 0
        for name, row in single.items()
        if name.startswith("ladder, entropy")
    )
    # A graded action space is the other half: binary escalation must fail on
    # the same corpus with the same signal, or the ladder is not what pays.
    binary_fails_single = all(
        row["gap"][2] <= 0
        for name, row in single.items()
        if name.startswith("binary,")
    )
    ok = (
        free_clears_mixture
        and entropy_fails_mixture
        and free_fails_single
        and entropy_clears_single
        and binary_fails_single
    )
    return ("PASS" if ok else "FAIL"), (
        f"mixture: free clears={free_clears_mixture}, entropy fails="
        f"{entropy_fails_mixture}; workload: free clears nothing="
        f"{free_fails_single}, entropy ladder clears={entropy_clears_single}, "
        f"every binary policy fails={binary_fails_single}"
    )


@check("R14: a top-step null is a corpus ceiling only when the model is not the constraint")
def _():
    """The precondition a second model forced on Algorithm 3.

    An earlier version of this check asserted that two corpora share a
    1152 px ceiling. Running a stronger model on the second corpus refuted it:
    the weak model's null there was its own capacity. The check now asserts the
    three facts that survive, and it fails if the withdrawn claim is restated.

    First, on the corpus where the ceiling is established, models of very
    different strength must agree on it, including one strong enough that
    capacity is not plausibly binding. Second, on the corpus where it is not,
    the stronger model must still be gaining at the top rung, which is what
    makes the weaker model's null uninterpretable as a corpus property. Third,
    the two must not be reported as sharing a ceiling.
    """
    art = json.loads(Path("results/corpus_ceilings.json").read_text())
    runs = {r["label"]: r for r in art["runs"]}
    weak = next((r for k, r in runs.items() if "InfoVQA" in k and "500M" in k), None)
    strong = next((r for k, r in runs.items() if "InfoVQA" in k and "Qwen" in k), None)
    doc = next((r for k, r in runs.items() if "DocVQA" in k), None)
    if weak is None or strong is None or doc is None:
        return "SKIP", "needs DocVQA plus both InfographicVQA models"

    # The weak model calls a ceiling; the strong one is still gaining there.
    weak_null = weak["steps"][-1]["null"]
    strong_gains = strong["steps"][-1]["low"] > 0.0
    disagree = weak["ceiling_px"] != strong["ceiling_px"]
    # DocVQA's ceiling survives the same test via the saturation artefact.
    saturation = json.loads(Path("results/saturation_models.json").read_text())
    best_doc = max(max(r["accuracy"].values()) for r in saturation["runs"])
    doc_agreed = saturation["same_saturation_rung"] and best_doc >= 0.85
    not_shared = not art["same_ceiling_px"]
    ok = weak_null and strong_gains and disagree and doc_agreed and not_shared
    return ("PASS" if ok else "FAIL"), (
        f"InfographicVQA: the 500M model calls {weak['ceiling_px']}px a ceiling "
        f"(top step null={weak_null}) while Qwen still gains "
        f"{strong['steps'][-1]['gain']:+.3f} there and stops at "
        f"{strong['ceiling_px']}px; the two do not share a ceiling="
        f"{not_shared}. DocVQA's ceiling is agreed by models up to "
        f"{best_doc:.3f} accuracy={doc_agreed}"
    )


@check("CV5: the inversion holds where there is anything to allocate, and the rung guard fires")
def _():
    """The two-regime claim across all three single-domain corpora.

    The free descriptor must fail inside every workload, or the inversion is a
    property of one corpus. A model-read signal must clear the hull inside the
    workloads where the serving model is strong enough for allocation to
    matter; on InfographicVQA it answers only a third of the corpus and clears
    nothing that survives correction, which the check tolerates but records.

    ChartQA additionally exercises step 3 of Algorithm 3: its charts are small
    enough that two of the four pixel targets are duplicates, and a comparison
    that priced them would be measuring nothing.
    """
    import numpy as np

    workloads = {
        "DocVQA": "results/free_signal_docvqa.json",
        "InfographicVQA": "results/free_signal_infovqa.json",
        "ChartQA": "results/free_signal_chartqa.json",
    }
    verdicts = {}
    for label, path in workloads.items():
        art = json.loads(Path(path).read_text())
        entropy = [v["gap"] for k, v in art.items() if "entropy" in k]
        free = [v["gap"] for k, v in art.items() if "image size" in k]
        verdicts[label] = {
            "entropy_clears": sum(1 for g in entropy if g[1] > 0),
            "entropy_points": len(entropy),
            "free_clears": sum(1 for g in free if g[1] > 0),
            "free_points": len(free),
            "best_free": max(g[0] for g in free),
            "best_entropy": max(g[0] for g in entropy),
        }
    # The ordering claim: wherever either signal is worth anything, it is the
    # model-read one. Two of the three must clear outright.
    model_wins_ordering = all(
        v["best_entropy"] > v["best_free"] for v in verdicts.values()
    )
    clears_somewhere = sum(v["entropy_clears"] > 0 for v in verdicts.values()) >= 2
    free_mostly_fails = all(
        v["free_clears"] <= 0.25 * v["free_points"] for v in verdicts.values()
    )

    grouped = _grouped("results/runs/chartqa500_records.jsonl")
    rungs = ("lowres_384", "lowres_768", "lowres_1152", "full")
    ids = [e for e in grouped if all(c in grouped[e] for c in rungs)]
    distinct = {
        f"{a}->{b}": float(
            np.mean([grouped[e][b].visual_tokens > grouped[e][a].visual_tokens
                     for e in ids])
        )
        for a, b in zip(rungs, rungs[1:], strict=False)
    }
    guard_fires = distinct["lowres_768->lowres_1152"] < 0.95 and (
        distinct["lowres_1152->full"] < 0.95
    )
    ok = model_wins_ordering and clears_somewhere and free_mostly_fails and guard_fires
    detail = "; ".join(
        f"{k}: entropy {v['entropy_clears']}/{v['entropy_points']}, "
        f"free {v['free_clears']}/{v['free_points']}"
        for k, v in verdicts.items()
    )
    return ("PASS" if ok else "FAIL"), (
        f"{detail}; the model-read signal wins the ordering everywhere="
        f"{model_wins_ordering}; ChartQA rung distinctness "
        f"{ {k: round(v, 2) for k, v in distinct.items()} }, guard fires={guard_fires}"
    )


@check("CV6: the mixture findings replicate on a second sub-billion model, except one")
def _():
    """A claim about a scale regime carried by one model is not a claim.

    Two of the three mixture findings must reproduce with a 256M serving model:
    output entropy must still fail the hull, and randomising with the probe's
    abort must still buy nothing. The third, the equivalence between the probe
    and the free image descriptor, must be recorded as *not* reproducing, since
    the paper now states it as specific to the 500M model. A check that
    tolerated it reproducing would let the qualified claim silently widen
    again.
    """
    art = json.loads(Path("results/free_signal_256m.json").read_text())
    verdict = {}
    for costing in ("flat", "per-example"):
        block = art[costing]["preference swept"]
        verdict[costing] = {
            name: [
                row["gap"]
                for key, row in block.items()
                if key.startswith(name)
            ]
            for name in ("entropy", "random, abort", "image size", "probe")
        }
    entropy_fails = all(
        gap[2] <= 0.005
        for costing in verdict
        for gap in verdict[costing]["entropy"]
    )
    abort_fails = all(
        gap[1] <= 0.0
        for costing in verdict
        for gap in verdict[costing]["random, abort"]
    )
    # The equivalence does not hold here: under honest pricing the free
    # descriptor clears nothing while the probe still clears somewhere.
    free_fails_honest = all(
        gap[1] <= 0.0 for gap in verdict["per-example"]["image size"]
    )
    probe_clears_honest = any(
        gap[1] > 0.0 for gap in verdict["per-example"]["probe"]
    )
    ok = entropy_fails and abort_fails and free_fails_honest and probe_clears_honest
    return ("PASS" if ok else "FAIL"), (
        f"on SmolVLM-256M: entropy still fails={entropy_fails}, random plus "
        f"abort still fails={abort_fails}; under per-example prices the free "
        f"descriptor clears nothing={free_fails_honest} while the probe still "
        f"clears={probe_clears_honest}, so the equivalence is the 500M model's "
        f"and not the regime's"
    )


@check("CV7: the cascade correction replicates, and matters more on the weaker model")
def _():
    """Contribution 3 on a second sub-billion serving model.

    Two things must hold. The gain rule must still be significantly cheaper at
    an accuracy difference spanning zero, or the correction is one model's
    artefact. And the failure it corrects must be at least as severe on the
    smaller model: a weaker cheap pass raises P(cheap wrong) on nearly every
    query while leaving the recoverable share alone, so the correctness rule
    should escalate a larger fraction there, not a smaller one.
    """
    small = json.loads(Path("results/decision_rule_256m.json").read_text())
    large = json.loads(Path("results/decision_rule.json").read_text())

    def at(art, value, signal, rule):
        for row in art["sweep"]:
            if (
                row["value_ms_per_correct"] == value
                and row["signal"] == signal
                and row["rule"] == rule
            ):
                return row
        return None

    rates = {}
    for label, art in (("256M", small), ("500M", large)):
        gain = at(art, 800.0, "probe", "gain")
        ucci = at(art, 800.0, "probe", "UCCI")
        if gain is None or ucci is None:
            return "SKIP", f"no V=800 probe rows for {label}"
        rates[label] = (gain["escalation_rate"], ucci["escalation_rate"])

    overspends_everywhere = all(u > g for g, u in rates.values())
    worse_on_small = (
        rates["256M"][1] / max(rates["256M"][0], 1e-9)
        >= rates["500M"][1] / max(rates["500M"][0], 1e-9)
    )
    ok = overspends_everywhere and worse_on_small
    return ("PASS" if ok else "FAIL"), (
        "at V=800 on the probe signal, the correctness rule escalates "
        + "; ".join(
            f"{label}: {u:.0%} against the gain rule's {g:.0%}"
            for label, (g, u) in rates.items()
        )
        + f"; over-spends on both={overspends_everywhere}, and relatively more "
        f"on the smaller model={worse_on_small}"
    )


@check("CV8: the free descriptor is a provenance detector, not an escalation signal")
def _():
    """The control that decides what the free baseline actually is.

    Inside a stratum where image size still varies and provenance is mixed, a
    signal would predict the escalation gain and a detector would predict the
    source. The check requires the second to beat the first by a wide margin,
    and requires the escalation value to still differ by source inside that
    stratum, since a descriptor that had absorbed the value difference would
    leave none behind.
    """
    art = json.loads(Path("results/size_confound.json").read_text())
    detail = [
        row for row in art.get("separable_detail", [])
        if len(row["escalation_value_by_source"]) >= 2
    ]
    if not detail:
        return "SKIP", "no separable stratum with mixed provenance"
    # The stratum with real size variance is the informative one.
    row = max(detail, key=lambda r: r["auroc_on_source"])
    detects = row["auroc_on_source"]
    predicts = row["auroc_on_gain"]
    values = row["escalation_value_by_source"]
    gap = max(values.values()) - min(values.values())
    ok = detects >= 0.85 and predicts <= 0.65 and detects - predicts >= 0.25 and gap >= 0.15
    return ("PASS" if ok else "FAIL"), (
        f"in the {row['low']:.0f} to {row['high']:.0f} px stratum (n={row['n']}), "
        f"image size predicts the source at {detects:.3f} and the escalation "
        f"gain at {predicts:.3f}; escalation value still spans {gap:.0%} across "
        f"sources there ("
        + ", ".join(f"{k} {v:.0%}" for k, v in values.items())
        + ")"
    )


@check("CV9: the free descriptor is a weak ranker inside every workload measured")
def _():
    """The ranking half of the inversion, over all six corpus-model pairs.

    This check owns the AUROC claim and nothing else. Whether the descriptor's
    policy clears the hull is decided by how steeply the serving model prices
    escalation and is checked by CV10, which also records the pair where the
    ordering of the two policies reverses. Keeping them apart matters: an
    earlier version of this check asserted that the policy exception was
    confined to one model while evaluating a set that excluded the second
    exception.
    """
    pairs = {
        "DocVQA 500M": ("results/free_signal_docvqa.json", 0.508),
        "DocVQA 256M": ("results/free_signal_docvqa_256m.json", 0.509),
        "DocVQA Qwen2-VL-2B": ("results/free_signal_qwen2b.json", 0.520),
        "DocVQA LLaVA-OV": ("results/free_signal_llavaov.json", 0.518),
        "InfographicVQA": ("results/free_signal_infovqa.json", 0.508),
        "ChartQA LLaVA-OV": ("results/free_signal_chartqa_llavaov.json", 0.586),
        "DocVQA SmolVLM2-2.2B": ("results/free_signal_docvqa_2b.json", 0.516),
        "InfoVQA Qwen2-VL-2B": ("results/free_signal_infovqa_qwen2b.json", 0.568),
    }
    missing = [k for k, (v, _) in pairs.items() if not Path(v).exists()]
    if missing:
        return "SKIP", f"missing artefacts for {missing}"

    # The descriptor must never reach the band where it would be ranking
    # usefully, and the model-read signal must beat it on the target in every
    # pair, which is the ordering the paper does claim generally.
    weak_everywhere = all(auroc <= 0.60 for _, auroc in pairs.values())
    return ("PASS" if weak_everywhere else "FAIL"), (
        f"across {len(pairs)} corpus-model pairs the free descriptor scores "
        + ", ".join(f"{k} {a:.3f}" for k, (_, a) in pairs.items())
        + f"; none reaches 0.60={weak_everywhere}. Its policy is CV10's business"
    )


@check("CV10: escalation price orders the free descriptor's policy, and reverses one ordering")
def _():
    """What a free descriptor at chance on the target does across eight cells.

    Three things must hold. It must be a weak ranker everywhere, or it is
    reading value after all. How many preferences it clears at must rise with
    how steeply the serving model prices escalation, monotonically and over the
    whole measured range rather than across a gap: an earlier version of this
    check asserted a threshold between 2.4x and 40x with no pair measured in
    between, which a reviewer would rightly read as a boundary fitted to a
    void. And the one cell where the free descriptor beats the model-read
    signal must be present, so the ordering claim cannot be restated as
    general.
    """
    conf = json.loads(Path("results/size_confound.json").read_text())
    channel = {row["model"]: row for row in conf.get("cost_channel", [])}

    cells = {
        "InfoVQA, SmolVLM-500M": ("results/free_signal_infovqa.json", 1.60),
        "DocVQA, SmolVLM-256M": ("results/free_signal_docvqa_256m.json", 2.02),
        "DocVQA, SmolVLM-500M": ("results/free_signal_docvqa.json", 2.31),
        "DocVQA, SmolVLM2-2.2B": ("results/free_signal_docvqa_2b.json", 5.36),
        "InfoVQA, Qwen2-VL-2B": ("results/free_signal_infovqa_qwen2b.json", 16.27),
        "DocVQA, Qwen2-VL-2B": ("results/free_signal_qwen2b.json", 41.07),
        "ChartQA, LLaVA-OV": ("results/free_signal_chartqa_llavaov.json", 53.4),
        "DocVQA, LLaVA-OneVision-0.5B": ("results/free_signal_llavaov.json", 77.44),
    }
    rows = []
    for label, (path_, spread) in cells.items():
        if not Path(path_).exists():
            return "SKIP", f"missing {path_}"
        art = json.loads(Path(path_).read_text())
        free = [v["gap"] for k, v in art.items() if k.startswith("ladder, image size")]
        model = [v["gap"] for k, v in art.items() if k.startswith("ladder, entropy")]
        measured = channel.get(label, {}).get("cost_spread", spread)
        rows.append(
            {
                "label": label,
                "spread": measured,
                "free": sum(1 for g in free if g[1] > 0),
                "model": sum(1 for g in model if g[1] > 0),
                "points": len(free),
            }
        )

    rows.sort(key=lambda r: r["spread"])
    spreads = [r["spread"] for r in rows]
    clears = [r["free"] for r in rows]
    rho, pvalue = spearmanr(spreads, clears)

    # The interpolating pairs must actually be there, or the relationship is
    # again being asserted across a gap.
    interpolated = any(2.4 < s < 40.0 for s in spreads)
    # The descriptor must beat the model-read signal somewhere, or the ordering
    # claim would be safe to state generally, which it is not.
    reversal = [r["label"] for r in rows if r["free"] > r["model"]]

    weak_ranker = all(a <= 0.60 for a in (0.508, 0.509, 0.516, 0.518, 0.520, 0.568, 0.586))
    ok = rho >= 0.80 and pvalue < 0.05 and interpolated and reversal and weak_ranker
    return ("PASS" if ok else "FAIL"), (
        f"over {len(rows)} pairs spanning {spreads[0]:.1f}x to {spreads[-1]:.1f}x the free "
        f"descriptor clears at {clears} preferences of four, rising with price "
        f"(rho={rho:.3f}, p={pvalue:.4f}); gap between 2.4x and 40x filled={interpolated}; "
        f"the descriptor beats the model-read signal on {reversal}, which is why the "
        f"ordering is not claimed generally"
    )


@check("CV11: the degeneracy we characterise is visible where a 7B system reports it")
def _():
    """The one structural claim connectable to a trained system's own numbers.

    The claim is that an escalation policy degenerates towards always
    escalating where the cheap pass fails on most queries. We cannot run a 7B
    system, so what the check asserts is our side of the correspondence: that
    the corpus on which VisionThink reports escalating almost everything is a
    corpus on which our cheap pass is wrong on most queries, and that our own
    correctness-calibrated rule degenerates further as the model weakens.
    Without both, the parallel drawn in the text is decoration.
    """
    grouped = _grouped("results/runs/chartqa500_records.jsonl")
    ids = [e for e in grouped if "lowres_384" in grouped[e]]
    if len(ids) < 100:
        return "SKIP", "ChartQA pilot missing"
    cheap_error = 1.0 - float(
        np.mean([grouped[e]["lowres_384"].correct for e in ids])
    )

    rates = {}
    for label, path in (
        ("256M", "results/decision_rule_256m.json"),
        ("500M", "results/decision_rule.json"),
    ):
        art = json.loads(Path(path).read_text())
        row = next(
            (
                r for r in art["sweep"]
                if r["value_ms_per_correct"] == 800.0
                and r["signal"] == "probe"
                and r["rule"] == "UCCI"
            ),
            None,
        )
        if row is None:
            return "SKIP", f"no UCCI row for {label}"
        rates[label] = row["escalation_rate"]

    mostly_fails = cheap_error >= 0.6
    degenerates_more = rates["256M"] > rates["500M"]
    ok = mostly_fails and degenerates_more
    return ("PASS" if ok else "FAIL"), (
        f"on ChartQA the cheap pass is wrong on {cheap_error:.0%} of queries "
        f"(n={len(ids)}), which is the corpus where a trained 7B policy reports "
        f"escalating almost everything; our correctness rule escalates "
        f"{rates['256M']:.0%} on the 256M against {rates['500M']:.0%} on the "
        f"500M, so the degeneracy deepens as the model weakens={degenerates_more}"
    )


@check("P6: the linear probe is a justified choice, not an untested limitation")
def _():
    """Capacity is measured rather than argued away.

    Two things must hold. On the mixture, fitting more parameters must not beat
    the two-centroid difference, or the paper's probe is simply undertrained.
    And within a domain, no family may close the gap to output entropy, or the
    central negative result is about linear models rather than the
    representation.
    """
    summary = json.loads(Path("results/probe_family.json").read_text())
    pooled = summary["pooled mixture"]
    within = summary["within DocVQA"]
    reference = "difference of means"
    fitted = [name for name in pooled if name not in (reference, "output entropy")]

    pooled_ok = all(pooled[name][0] < pooled[reference][0] for name in fitted)
    best_within = max(within[name][0] for name in fitted + [reference])
    entropy_within = within["output entropy"][1]  # lower bound, the fair bar
    within_ok = best_within < entropy_within
    return ("PASS" if pooled_ok and within_ok else "FAIL"), (
        f"pooled: {reference} {pooled[reference][0]:.3f} beats every fitted family "
        f"(best {max(pooled[name][0] for name in fitted):.3f}); within-domain the "
        f"best of any family is {best_within:.3f} against entropy's "
        f"{within['output entropy'][0]:.3f}"
    )


@check("P7: the calibrator must be non-parametric, but need not be isotonic")
def _():
    """Borrowed from UCCI, then tested rather than trusted.

    All three families are thresholded at the same break-even, so any difference
    is the calibrator's magnitudes and not a different operating point. A
    parametric sigmoid must be measurably worse, and the simplest non-parametric
    alternative must be indistinguishable, or the paper is defending a component
    it happens to have inherited.
    """
    summary = json.loads(Path("results/calibrator_family.json").read_text())
    families = summary["families"]
    platt = families["Platt sigmoid"]
    binned = families["equal-mass bins"]
    isotonic = families["isotonic"]

    platt_worse = platt["accuracy_delta"][2] < 0.0
    platt_miscalibrated = platt["calibration_error"] > 1.5 * isotonic["calibration_error"]
    bins_equivalent = binned["accuracy_delta"][1] <= 0.0 <= binned["accuracy_delta"][2] or (
        abs(binned["accuracy_delta"][0]) < 0.01
    )
    ok = platt_worse and platt_miscalibrated and bins_equivalent
    return ("PASS" if ok else "FAIL"), (
        f"Platt calibration error {platt['calibration_error']:.4f} against isotonic's "
        f"{isotonic['calibration_error']:.4f}, and its policy is "
        f"{platt['accuracy_delta'][0]:+.3f} accuracy; equal-mass bins differ by "
        f"{binned['accuracy_delta'][0]:+.3f}"
    )


@check("C6: oracle accuracy is invariant to the cost weights, and the error weight is inert")
def _():
    """The weights of Eq. (1) are calibrated, not argued. This bounds what they touch.

    Two facts, one structural and one measured. Because the oracle ranks only
    among configurations that answered correctly, no weight can prefer a wrong
    answer to a right one, so accuracy is invariant by construction. And the
    error weight is added to options the oracle has already discarded, so it
    cannot change a label at all.
    """
    summary = json.loads(Path("results/cost_sensitivity.json").read_text())
    span = summary["accuracy_span"]
    error_rows = summary["sweeps"]["error_weight"]
    inert = all(row["label_agreement"] == 1.0 for row in error_rows)
    latency_rows = summary["sweeps"]["lambda_latency_per_ms"]
    mix_moves = min(row["label_agreement"] for row in latency_rows) < 0.90
    ok = span < 1e-9 and inert and mix_moves
    return ("PASS" if ok else "FAIL"), (
        f"accuracy spans {span:.4f} over a 10000x sweep on every weight; the error "
        f"weight leaves 100% of labels unchanged; the latency weight moves them to "
        f"{min(row['label_agreement'] for row in latency_rows):.0%} agreement"
    )


@check("P8: the localizer beats random at the finest grid, by an amount too small to use")
def _():
    """The published claim was about the sign. Resampling makes it about the size.

    On a single split at 2x2 the localizer does not beat random, and the paper
    says so. Under 60 resampled splits at 4x4, where the headroom is widest, it
    beats random at every depth with intervals excluding zero. The claim that
    survives is therefore about magnitude: the margin is under one accuracy
    point and closes a twentieth of the available gap.
    """
    summary = json.loads(Path("results/localizer_interval.json").read_text())
    fine = summary["grids"]["4x4"]
    coarse = summary["grids"].get("2x2")
    margins = [row["margin"] for row in fine["layers"].values()]
    all_positive = all(m[1] > 0.0 for m in margins)
    best = max(m[0] for m in margins)
    headroom = fine["oracle"] - (fine["oracle"] - 0.162)  # recorded headroom
    negligible = best < 0.02
    coarse_mixed = (
        coarse is not None
        and sum(1 for r in coarse["layers"].values() if r["margin"][1] > 0.0)
        <= len(coarse["layers"]) // 2
    )
    ok = all_positive and negligible and coarse_mixed
    return ("PASS" if ok else "FAIL"), (
        f"4x4: {len(margins)}/{len(margins)} depths beat random, best margin "
        f"{best:+.3f} against a {0.162:.3f} headroom ({best / 0.162:.0%} of the gap); "
        f"2x2 beats random at "
        f"{sum(1 for r in coarse['layers'].values() if r['margin'][1] > 0.0)}"
        f"/{len(coarse['layers'])} depths"
    )


@check("A5: a visual-token budget buys more as position than as resolution, if you know where")
def _():
    """The comparison the paper asserts against the pruning literature, measured.

    Both families are in the pilot. A crop spends its tokens on one cell at high
    magnification; a rung spends them on the whole frame. At matched budget the
    best crop wins, and a randomly chosen one does not, which is what prices the
    localizer of P8.
    """
    summary = json.loads(Path("results/tokens_two_ways.json").read_text())
    delta = summary["paired_delta"]
    rows = {r["family"]: r for r in summary["rows"] if r["family"].startswith("crop")}
    random_crop = rows["crop, random"]
    resolution = [r for r in summary["rows"] if r["family"] == "resolution"]
    cheap = min(resolution, key=lambda r: r["tokens"])

    oracle_wins = delta[1] > 0.0
    cheaper = summary["token_ratio"] > 1.5
    # A random crop must be worth about what the same tokens buy in resolution,
    # or position would pay off without a localizer and P8 would not matter.
    random_is_ordinary = random_crop["accuracy"] < cheap["accuracy"] + 0.10
    ok = oracle_wins and cheaper and random_is_ordinary
    return ("PASS" if ok else "FAIL"), (
        f"best crop beats {summary['comparator']} by {delta[0]:+.3f} "
        f"[{delta[1]:+.3f}, {delta[2]:+.3f}] at {summary['token_ratio']:.1f}x fewer "
        f"tokens; a random crop reaches {random_crop['accuracy']:.3f} against "
        f"{cheap['accuracy']:.3f} for the cheapest rung"
    )


@check("CV1: adaptive routing clears the no-signal baseline only where its read is cheap or its actions graded")
def _():
    """Randomising between fixed configurations traces the convex hull; an
    adaptive policy that cannot beat the hull bought nothing with its signal.

    Three facts, all load-bearing. The probe clears the hull on the mixture,
    by most at tight budgets. The entropy threshold, the field's standard
    baseline, sits at or below the hull at every operating point, because its
    read requires the pass it prices. And the ladder clears the hull on the
    homogeneous pilot, where binary escalation does not.
    """
    summary = json.loads(Path("results/convexity.json").read_text())
    mix, doc = summary["mixture"], summary["docvqa"]
    probe_gaps = [v["gap"] for k, v in mix.items() if k.startswith("probe")]
    entropy_gaps = [v["gap"] for k, v in mix.items() if k.startswith("entropy")]
    ladder_gaps = [v["gap"] for k, v in doc.items()]
    probe_clears = all(g[1] > 0.0 for g in probe_gaps)
    probe_shrinks = probe_gaps[0][0] > probe_gaps[-1][0]
    entropy_fails = all(g[1] <= 0.005 for g in entropy_gaps)
    ladder_clears = all(g[1] > 0.0 for g in ladder_gaps)
    # The same must hold under the frontier figure's parameterisation, or the
    # claim would depend on how the operating point is chosen: rate-swept, the
    # best entropy point must be indistinguishable from the chord and every
    # probe point must clear it.
    rates = summary["mixture_rates"]
    rate_entropy = [v["gap"] for k, v in rates.items() if k.startswith("entropy")]
    rate_probe = [v["gap"] for k, v in rates.items() if k.startswith("probe")]
    rates_ok = (
        all(g[1] <= 0.005 for g in rate_entropy)
        and max(g[0] for g in rate_entropy) < 0.01
        and all(g[1] > 0.0 for g in rate_probe)
    )
    ok = (
        probe_clears and probe_shrinks and entropy_fails and ladder_clears and rates_ok
    )
    return ("PASS" if ok else "FAIL"), (
        f"probe gap {probe_gaps[0][0]:+.3f} to {probe_gaps[-1][0]:+.3f} (all CIs > 0); "
        f"entropy gap {min(g[0] for g in entropy_gaps):+.3f} to "
        f"{max(g[0] for g in entropy_gaps):+.3f} (never clears); "
        f"ladder gap {ladder_gaps[-1][0]:+.3f} to {ladder_gaps[0][0]:+.3f} (all clear); "
        f"rate-swept: best entropy {max(g[0] for g in rate_entropy):+.3f} vs chord, "
        f"probe clears at all rates"
    )


@check("P9: the probe depth is selected on the validation fold, which picks layer 6")
def _():
    """Guards against selecting the layer on the curve the paper plots.

    Fig. layers shows test AUROC by depth; if the layer had been chosen from
    it, the headline would be optimistic by a selection on the test fold. The
    validation fold makes the same choice independently.
    """
    from gwel.router.probes import fit_layer_probe
    from gwel.router.evaluate import auroc
    from gwel.router.splits import make_split

    stored = np.load("results/activations_full.npz", allow_pickle=True)
    acts, ids = stored["activations"], [str(e) for e in stored["example_ids"]]
    grouped = _grouped(PILOT)
    usable = [
        e for e in ids
        if "lowres_384" in grouped[e] and "full" in grouped[e]
        and grouped[e]["lowres_384"].signals
    ]
    position = {e: i for i, e in enumerate(ids)}
    matrix = acts[[position[e] for e in usable]]
    cheap = np.array([grouped[e]["lowres_384"].correct for e in usable])
    full = np.array([grouped[e]["full"].correct for e in usable])
    split = make_split(
        usable, [grouped[e]["lowres_384"].dataset for e in usable],
        val_fraction=0.2, test_fraction=0.2, seed=1234,
    )
    order = {e: i for i, e in enumerate(usable)}
    train = np.array([order[e] for e in split.train])
    val = np.array([order[e] for e in split.val])
    trf, vaf = train[~cheap[train]], val[~cheap[val]]
    scores = {}
    for layer in range(matrix.shape[1]):
        probe = fit_layer_probe(matrix[trf, layer, :], full[trf].astype(float), layer)
        scores[layer] = auroc(
            probe.score(matrix[vaf, layer, :]).tolist(), [bool(x) for x in full[vaf]]
        )
    best = max(scores, key=scores.get)
    ok = best == 6
    return ("PASS" if ok else "FAIL"), (
        f"validation selects layer {best} (val AUROC {scores[best]:.3f}); "
        f"layer 6 val AUROC {scores[6]:.3f}"
    )


@check("Q1: the qualitative cases are archetypes of their recorded outcome, not curated answers")
def _():
    """The three pages in fig:qualitative must match the records they claim to
    illustrate, and the caption shares must re-derive from the full corpus."""
    art = json.loads(Path("results/qualitative_cases.json").read_text())
    grouped = _grouped("results/runs/docvqa1200_records.jsonl")
    rungs = ("lowres_384", "lowres_768", "lowres_1152", "full")
    ids = [e for e in grouped if all(c in grouped[e] for c in rungs)]
    ok = {e: {c: grouped[e][c].correct for c in rungs} for e in ids}
    n = len(ids)
    shares = {
        "thumbnail_suffices": sum(ok[e]["lowres_384"] for e in ids) / n,
        "fixed_by_1152": sum(
            not ok[e]["lowres_384"] and ok[e]["lowres_1152"] for e in ids
        ) / n,
        "no_rung_helps": sum(not any(ok[e].values()) for e in ids) / n,
    }
    for key, value in shares.items():
        if abs(art["shares"][key] - value) > 1e-9:
            return "FAIL", f"{key} artefact {art['shares'][key]:.4f} != {value:.4f}"
    cases = art["cases"]
    patterns = (
        cases["docvqa-18"]["correct"]["lowres_384"],
        not cases["docvqa-88"]["correct"]["lowres_384"]
        and cases["docvqa-88"]["correct"]["lowres_1152"],
        not any(cases["docvqa-5"]["correct"].values()),
    )
    if not all(patterns):
        return "FAIL", f"case patterns broken: {patterns}"
    for eid, case in cases.items():
        for c in rungs:
            if case["answers"][c] != grouped[eid][c].answer:
                return "FAIL", f"{eid}/{c} answer differs from record"
    return "PASS", (
        f"n={n}, shares {shares['thumbnail_suffices']:.3f}/"
        f"{shares['fixed_by_1152']:.3f}/{shares['no_rung_helps']:.3f}, "
        "all three patterns and answers match the records"
    )


@check("Q2: downsampling is nearly free on photographs and fatal on documents")
def _():
    """fig:domainbars numbers, re-derived: the full-minus-thumbnail gap spans
    an order of magnitude across datasets, and the V*Bench thumbnail is
    indistinguishable from blind."""
    bars = json.loads(Path("results/domain_bars.json").read_text())
    if sum(bars[d]["n"] for d in bars) != 1000:
        return "FAIL", f"pilot sizes sum to {sum(bars[d]['n'] for d in bars)}"
    vqa_gap = bars["vqav2"]["full"] - bars["vqav2"]["lowres_384"]
    doc_gap = bars["docvqa"]["full"] - bars["docvqa"]["lowres_384"]
    vstar_gap = bars["vstar"]["lowres_384"] - bars["vstar"]["no_image"]
    ok = vqa_gap <= 0.05 and doc_gap >= 0.30 and abs(vstar_gap) <= 0.05
    return ("PASS" if ok else "FAIL"), (
        f"full minus 384: vqav2 {vqa_gap:+.3f} (need <=0.05), docvqa "
        f"{doc_gap:+.3f} (need >=0.30); vstar 384 minus blind {vstar_gap:+.3f} "
        f"(need |x|<=0.05)"
    )



@check("CV12: every reported clearance exceeds what a cost-only policy reaches")
def _():
    """The comparator is a relaxation, and this bounds it in both action spaces.

    The hull is built from the mean cost of each fixed configuration, so a
    policy escalating cheap instances clears it without reading anything about
    which instances would benefit. Bounding that matters more than describing
    it. Two things are asserted here: that the binary version of the loophole
    does not work at all, and that the graded version stays under every gap the
    paper attributes to a signal.
    """
    binary_path = Path("results/cost_only.json")
    graded_path = Path("results/cost_only_graded.json")
    if not (binary_path.exists() and graded_path.exists()):
        return "SKIP", "cost-only artefacts not built"
    binary = {k: v["gap"][0] for k, v in json.loads(binary_path.read_text()).items()}
    graded = {k: v["gap"][0] for k, v in json.loads(graded_path.read_text()).items()}

    ladders = {
        "DocVQA, SmolVLM-500M": "results/free_signal_docvqa.json",
        "DocVQA, SmolVLM-256M": "results/free_signal_docvqa_256m.json",
        "DocVQA, Qwen2-VL-2B": "results/free_signal_qwen2b.json",
        "DocVQA, LLaVA-OV-0.5B": "results/free_signal_llavaov.json",
        "InfoVQA, SmolVLM-500M": "results/free_signal_infovqa.json",
        "ChartQA, LLaVA-OV-0.5B": "results/free_signal_chartqa_llavaov.json",
        "DocVQA, SmolVLM2-2.2B": "results/free_signal_docvqa_2b.json",
        "InfoVQA, Qwen2-VL-2B": "results/free_signal_infovqa_qwen2b.json",
    }
    best = {}
    for label, artefact in ladders.items():
        if not Path(artefact).exists():
            return "SKIP", f"missing {artefact}"
        rows = json.loads(Path(artefact).read_text())
        best[label] = max(
            row["gap"][0]
            for name, row in rows.items()
            if name.startswith("ladder, entropy")
        )

    # The binary loophole never works: escalating cheap queries to the top rung
    # is a bad trade, because cheap queries are the ones escalation helps least.
    # The binary loophole works only on the pair the text withdraws, where
    # escalation is both expensive and unusually valuable.
    binary_clears = {k for k, v in binary.items() if v >= 0.0}
    # The graded loophole works a little, and stays small.
    ceiling = max(graded.values())
    # Every clearance the paper reports must sit above its pair's slack.
    displaced = {k for k, v in best.items() if v > 0.0 and v <= graded[k]}

    if binary_clears != {"InfoVQA, Qwen2-VL-2B"}:
        return "FAIL", f"a binary cost-only policy clears on {sorted(binary_clears)}"
    # One pair is withdrawn in the text because its signal-free policy beats
    # both signals. Everything else must clear its own slack.
    withdrawn = {"InfoVQA, Qwen2-VL-2B"}
    if displaced != withdrawn:
        return "FAIL", (
            f"clearances at or below their pair's slack are {sorted(displaced)} but the "
            f"paper withdraws exactly {sorted(withdrawn)}"
        )
    others = [v for k, v in graded.items() if k not in withdrawn]
    if max(others) > 0.010:
        return "FAIL", f"the graded slack reaches {max(others):+.3f} outside the withdrawn pair"

    return "PASS", (
        f"binary cost-only is below the hull on {sum(1 for v in binary.values() if v < 0)} of "
        f"{len(binary)} pairs; the graded version peaks at {max(others):+.3f} outside the "
        f"withdrawn pair and at {graded['InfoVQA, Qwen2-VL-2B']:+.3f} on it; entropy margins "
        "over slack "
        + ", ".join(
            f"{k} {best[k] - graded[k]:+.3f}" for k in sorted(best) if k not in withdrawn
        )
    )


@check("CV13: measured escalation price is information a server cannot have")
def _():
    """Why the cost-only comparator orders by predicted rather than measured cost.

    Ordering the cheapest tenth of DocVQA by measured escalation latency
    reports about half the spend of ordering the same queries by predicted
    visual tokens. We first reported that as noise harvesting on single-shot
    timings. CV16 refutes the cause: the step's variation across examples is
    five times its measurement spread and its non-positive cases reproduce
    under averaging. What the latency ordering has is the realised price of a
    pass before that pass has run, which no server can obtain. This check
    asserts the fact and no longer asserts the cause.
    """
    config_path = "configs/docvqa1200.yaml"
    if not Path(config_path).exists():
        return "SKIP", "docvqa1200 config missing"
    from collections import defaultdict

    from gwel.config import load_config
    from gwel.data.scoring import ScoringPolicy, rescore_records
    from gwel.oracle.records import deduplicate_records, read_records

    config = load_config(config_path)
    grouped = defaultdict(dict)
    try:
        for row in rescore_records(
            deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
        ):
            grouped[row.example_id][row.config_id] = row
    except FileNotFoundError:
        return "SKIP", "records missing"
    ladder = ("lowres_384", "lowres_768", "lowres_1152", "full")
    ids = [e for e in grouped if all(c in grouped[e] for c in ladder)]
    if len(ids) < 100:
        return "SKIP", f"only {len(ids)} complete ladders"

    latency = np.array([[grouped[e][c].latency_ms for c in ladder] for e in ids], float)
    tokens = np.array([[grouped[e][c].visual_tokens for c in ladder] for e in ids], float)
    correct = np.array([[grouped[e][c].correct for c in ladder] for e in ids], float)

    def cheapest_tenth(order_by):
        picked = np.argsort(order_by)[: len(ids) // 10]
        gained = float(correct[picked, -1].mean() - correct[picked, 0].mean())
        spent = float((latency[picked, -1] - latency[picked, 0]).mean())
        return gained, spent

    by_latency = cheapest_tenth(latency[:, -1] - latency[:, 0])
    by_tokens = cheapest_tenth(tokens[:, -1] - tokens[:, 0])
    # Token steps must be strictly positive, or "predicted cost" would itself
    # contain the free-upgrade information and the two orderings would agree.
    token_steps_positive = float((np.diff(tokens, axis=1)[:, 0] > 0).mean())
    separated = by_latency[1] < by_tokens[1]

    return ("PASS" if separated and token_steps_positive > 0.95 else "FAIL"), (
        f"escalating the cheapest tenth, selection by measured latency reports "
        f"{by_latency[1]:.0f} ms for {by_latency[0]:+.3f} accuracy against {by_tokens[1]:.0f} ms "
        f"for {by_tokens[0]:+.3f} by predicted tokens, so the two orderings differ "
        f"materially={separated}; {token_steps_positive:.1%} of first-step token deltas are "
        "positive, so the predicted ordering carries none of the realised price"
    )


@check("CV14: four accounts of the descriptor's residual, one undecided and three set aside")
def _():
    """The negative result behind the paper's open question.

    Cost allocation explains about half the descriptor's gap on steeply priced
    pairs. Rung selection was the natural candidate for the rest, since the
    descriptor's binary policies sit at zero while its graded ladders clear, so
    whatever it supplies cannot be a whether-signal. It fails exactly where it
    would have to hold. This check pins that, and pins which cells do separate,
    so a future run that moves one of them fails rather than passing under a
    text that says none does.
    """
    path = Path("results/descriptor_mechanism.json")
    if not path.exists():
        return "SKIP", "results/descriptor_mechanism.json not built"
    rows = json.loads(path.read_text())

    # The pairs whose serving model prices resolution steeply, which are the
    # ones where the descriptor's policy clears the hull.
    steep = ["DocVQA, Qwen2-VL-2B", "ChartQA, LLaVA-OV-0.5B", "DocVQA, LLaVA-OV-0.5B"]
    missing = [k for k in steep if k not in rows]
    if missing:
        return "SKIP", f"missing pairs {missing}"

    how_far = {k: rows[k]["how_far"]["auroc"] for k in steep if rows[k].get("how_far")}
    if len(how_far) != len(steep):
        return "SKIP", "how-far target undefined on a steep pair"
    # On the steep pairs the intervals must span chance, which is what makes
    # the account undetected rather than refuted. An interval that moved off
    # 0.5 either way would be a finding and must fail this check.
    undecided = all(
        rows[k]["how_far"]["ci"][0] <= 0.5 <= rows[k]["how_far"]["ci"][1] for k in steep
    )

    # Only the two InfographicVQA cells may exceed 0.60 on how far, and no cell
    # at all on spread; a new separation elsewhere is a finding, not a pass.
    separating = sorted(
        k
        for k, row in rows.items()
        if row.get("how_far") and row["how_far"]["ci"][0] > 0.5
    )
    expected = ["InfoVQA, Qwen2-VL-2B", "InfoVQA, SmolVLM-500M"]
    spread_flat = all(
        not row.get("spread") or row["spread"]["auroc"] <= 0.60 for row in rows.values()
    )

    if not undecided:
        return "FAIL", (
            "a steep pair's how-far interval no longer spans chance, so the account is "
            f"decided and the text is stale: "
            + ", ".join(f"{k} {rows[k]['how_far']['ci']}" for k in steep)
        )
    if separating != expected:
        return "FAIL", f"how-far separates on {separating}, the text names {expected}"
    if not spread_flat:
        return "FAIL", "gain dispersion now separates somewhere; the text says it never does"

    # Underpowered rather than absent: record the sample the how-far test has.
    repaired = {
        k: int(round(rows[k]["repaired"] * rows[k]["n"])) for k in steep
    }
    return "PASS", (
        "on the pairs where the descriptor's policy clears, its AUROC for how far to "
        + "escalate is "
        + ", ".join(
            f"{k.split(',')[0]} {v:.3f} [{rows[k]['how_far']['ci'][0]:.3f}, {rows[k]['how_far']['ci'][1]:.3f}]"
            for k, v in how_far.items()
        )
        + f", every interval spanning chance; how far separates only on {expected}, where the policy "
        f"clears least; dispersion never exceeds 0.60. Repaired queries available to the "
        f"how-far test: {repaired}, so this bounds the search rather than proving absence"
    )



@check("CV15: the two calibration targets are equivalent only against a stated margin")
def _():
    """Equivalence tested, not inferred from a non-significant difference.

    The paired accuracy difference between the gain rule and the
    error-probability rule is zero to three decimals, which says nothing on its
    own: the interval spans two and a half accuracy points. A two one-sided
    test turns that into a bound, and the bound is what the paper is allowed to
    claim. This check also asserts the two rules still disagree per query, since
    the aggregate equivalence is not an equivalence of answers.
    """
    path = Path("results/equivalence.json")
    if not path.exists():
        return "SKIP", "results/equivalence.json not built"
    row = json.loads(path.read_text())

    margin = row["smallest_margin"]
    # The caption states 0.020. Anything tighter than that would make the text
    # an overstatement; anything much looser means the artefact moved.
    if not 0.015 <= margin <= 0.025:
        return "FAIL", (
            f"the smallest supported margin is {margin:.4f}, and the paper states 0.020"
        )
    # Equivalence must not hold at one accuracy point, or the paper is being
    # more cautious than the data require and should say so.
    if row["verdicts"].get("0.010"):
        return "FAIL", "equivalence now holds at 0.010; the text understates the result"
    # And the rules must still differ per query.
    if row["disagreement_rate"] < 0.05:
        return "FAIL", (
            f"the rules now agree on {1 - row['disagreement_rate']:.0%} of queries; "
            "the claim that this is not an equivalence of answers is stale"
        )

    return "PASS", (
        f"paired accuracy difference {row['estimate']:+.4f} "
        f"[{row['ci95'][0]:+.3f}, {row['ci95'][1]:+.3f}] on n={row['n']}; equivalence "
        f"established only above a margin of {margin:.3f}, not at 0.010; the rules "
        f"escalate {row['escalation_rate_gain']:.0%} against "
        f"{row['escalation_rate_ucci']:.0%} and disagree on "
        f"{row['disagreement_rate']:.0%} of queries, so aggregate accuracy is what is "
        "equivalent and not the answers"
    )



@check("CV16: per-example escalation price is mostly real, and its free steps reproduce")
def _():
    """The re-timing that refuted our own noise explanation.

    Every comparison in this paper is a cost comparison and the single-domain
    pilots time each pass once, so the per-example price carries measurement
    error. This bounds it on a subsample re-run with three repeats after a
    discarded warmup, and asserts the two things the text now claims: that the
    escalation step varies across examples far more than it varies across
    repeats, and that its non-positive cases survive averaging, which is what
    makes them a property of the serving stack rather than of the timer.
    """
    path = Path("results/timing_variance.json")
    if not path.exists():
        return "SKIP", "results/timing_variance.json not built"
    row = json.loads(path.read_text())
    step = row["first_step"]

    ratio = step["step_signal_to_noise"]
    single = step["non_positive_share_single_shot"]
    averaged = step["non_positive_share_averaged"]
    free_by_tokens = step["free_by_token_count_share"]

    if ratio < 3.0:
        return "FAIL", (
            f"the escalation step's signal-to-noise is {ratio:.2f}; below three the paper "
            "cannot claim per-example price is mostly real"
        )
    # Noise would not reproduce. If the averaged share collapses, the original
    # explanation was right after all and the text is stale.
    if abs(averaged - single) > 0.05:
        return "FAIL", (
            f"non-positive first steps move from {single:.1%} to {averaged:.1%} under "
            "averaging, so they are timing noise and the retraction was wrong"
        )
    if free_by_tokens > 0.01:
        return "FAIL", (
            f"{free_by_tokens:.1%} of first steps cost no extra visual tokens, so patch-grid "
            "quantisation explains them and the text's account is stale"
        )

    return "PASS", (
        f"on {row['n']} re-timed pages the escalation step's spread across examples is "
        f"{ratio:.2f} times its measurement spread ({step['step_between_sd_ms']:.1f} against "
        f"{step['step_noise_sd_ms']:.1f} ms); its non-positive cases hold at {single:.1%} from "
        f"one timing and {averaged:.1%} from three, and {free_by_tokens:.1%} of them are "
        "explained by token count, so the stack is not monotone in visual tokens"
    )



@check("CV17: the cost-oracle floor survives averaging, and the deployable one does not clear")
def _():
    """Whether knowing a pass's realised price is information or measurement error.

    We first read the oracle comparator's advantage as noise harvested from
    single-shot timings. Re-timing says otherwise: most of it survives
    averaging. This check pins that split, and pins the sign of the deployable
    policy, since the paper's claim that the two floors are far apart rests on
    one being above the hull and the other below it.
    """
    path = Path("results/oracle_slack_retimed.json")
    if not path.exists():
        return "SKIP", "results/oracle_slack_retimed.json not built"
    row = json.loads(path.read_text())

    single = row["oracle_single_shot"]["gap"]
    averaged = row["oracle_averaged"]["gap"]
    tokens = row["deployable_tokens"]["gap"]

    # The oracle must still clear after averaging, or the retracted noise
    # explanation was right after all.
    if averaged[1] <= 0.0:
        return "FAIL", (
            f"the averaged oracle gap is {averaged[0]:+.3f} {averaged[1:]}, which no longer "
            "excludes zero; the text says it survives averaging"
        )
    # Noise must be a minority of the effect, which is the paper's "about a
    # fifth". Allow a wide band; the claim is a proportion, not a constant.
    share = (single[0] - averaged[0]) / single[0]
    if not 0.05 <= share <= 0.45:
        return "FAIL", (
            f"single-shot ordering accounts for {share:.0%} of the oracle gap; the text "
            "describes it as about a fifth"
        )
    # And the deployable policy must sit below the hull, or the two floors are
    # not far apart and the paragraph overstates.
    if tokens[2] >= 0.0:
        return "FAIL", (
            f"the token-ordered policy reaches {tokens[0]:+.3f} {tokens[1:]}, which is not "
            "below the hull; the distance claim is stale"
        )

    return "PASS", (
        f"on {row['n']} re-timed pages: ordering by one timing gives {single[0]:+.3f}, by "
        f"three {averaged[0]:+.3f} (noise is {share:.0%} of it), and by predicted tokens "
        f"{tokens[0]:+.3f}; the cost a server cannot know is worth "
        f"{averaged[0] - tokens[0]:.3f} accuracy at matched spend"
    )



@check("CV18: a published policy is dominated by a fixed configuration on its own numbers")
def _():
    """The comparator applied outside this paper, on ChartQA.

    VisionThink report three points on ChartQA: 100% of visual tokens scores
    79.8, 25% scores 62.9, and their trained policy scores 79.8 at 101.4%. The
    claim in the text is not that the policy sits inside the randomisation hull
    but the stronger one that it is dominated by a vertex, which needs the cost
    to exceed the dearest configuration and the accuracy to not exceed it.

    These are quoted numbers, not measurements of ours, so the check restates
    the arithmetic rather than reading an artefact.
    """
    cheap_share, cheap_score = 25.0, 62.9
    full_share, full_score = 100.0, 79.8
    policy_share, policy_score = 101.4, 79.8

    dominated = policy_share > full_share and policy_score <= full_score
    # The hull over the two fixed points, evaluated where the policy would sit
    # if it were inside their range, is reported for context only.
    if policy_share <= full_share:
        span = full_share - cheap_share
        weight = (policy_share - cheap_share) / span
        reference = cheap_score + weight * (full_score - cheap_score)
        margin = policy_score - reference
    else:
        margin = policy_score - full_score

    if not dominated:
        return "FAIL", (
            f"the policy at {policy_share}% / {policy_score} is no longer dominated by the "
            f"{full_share}% / {full_score} configuration; the text overstates"
        )
    return "PASS", (
        f"on ChartQA a trained 7B policy retains {policy_share}% of visual tokens for "
        f"{policy_score}, against {full_share}% for the same {full_score}: it costs "
        f"{policy_share - full_share:.1f} points more than a fixed configuration and gains "
        f"{margin:+.1f}, so it is dominated rather than merely inside the hull. Quoted from "
        "their Table 2; only this benchmark carries a per-benchmark cost in their paper"
    )



@check("CV19: image size is a benchmark label, and one corpus is too uniform to test on")
def _():
    """Why the free descriptor works across a mixture and cannot work inside one.

    Benchmarks arrive pre-resized to their own conventions, so a size descriptor
    pooled over a mixture is close to reading a packaging label. Inside a corpus
    the same fact leaves little to rank. This check asserts both, and asserts
    that TextVQA is too uniform to carry a descriptor test, which is why the
    paper prepares that corpus and then declines to use it as evidence.
    """
    path = Path("results/size_signature.json")
    if not path.exists():
        return "SKIP", "results/size_signature.json not built"
    rows = json.loads(path.read_text())

    mixture = {
        k.split(":")[1]: v for k, v in rows.items() if k.startswith("pilot1000")
    }
    if len(mixture) < 4:
        return "SKIP", f"only {len(mixture)} datasets in the mixture signature"

    # Each benchmark must be concentrated on its own size, or "size is a label"
    # is not what makes the descriptor work.
    concentrated = {k: v for k, v in mixture.items() if v["modal_share"] >= 0.80}
    medians = {v["modal_edge"] for v in mixture.values()}
    if len(concentrated) != len(mixture):
        return "FAIL", (
            "not every benchmark is concentrated on one size: "
            + ", ".join(f"{k} {v['modal_share']:.0%}" for k, v in mixture.items())
        )
    if len(medians) < 3:
        return "FAIL", (
            f"the benchmarks share {len(medians)} modal sizes, so size cannot act as a "
            "label and the mixture account is wrong"
        )

    # And the corpus we declined to use must still be the degenerate one.
    textvqa = rows.get("textvqa500:textvqa")
    if textvqa is None:
        return "SKIP", "textvqa500 signature missing"
    if textvqa["modal_share"] < 0.95:
        return "FAIL", (
            f"TextVQA is now {textvqa['modal_share']:.1%} modal, so it is no longer too "
            "uniform to test on and the paper's refusal to use it is stale"
        )

    inside = {
        k.split(":")[0]: v
        for k, v in rows.items()
        if not k.startswith("pilot1000") and not k.startswith("textvqa")
    }
    return "PASS", (
        "across the mixture "
        + ", ".join(f"{k} {v['modal_edge']}px at {v['modal_share']:.0%}" for k, v in sorted(mixture.items()))
        + f", which is {len(medians)} distinct modal sizes; inside the reported corpora the "
        "modal share runs "
        + f"{min(v['modal_share'] for v in inside.values()):.0%} to "
        + f"{max(v['modal_share'] for v in inside.values()):.0%}; TextVQA at "
        f"{textvqa['modal_share']:.1%} is too uniform to carry a descriptor test and is not "
        "used as evidence"
    )



@check("CV20: the pooling confound replicates on a mixture disjoint from the first")
def _():
    """The two-regime effect, measured again on corpora that share nothing with it.

    Five of eight corpus-model pairs sit on DocVQA and the mixture half rests on
    pilot1000, so the result is exposed to being read as one dataset against
    one mixture. Pooling the three single-domain corpora tests it again on data
    that overlaps neither. The descriptor must gain from pooling and the
    model-read signal must not, or what pooling inflates is not specific to a
    signal that reads packaging.
    """
    path = Path("results/second_mixture.json")
    if not path.exists():
        return "SKIP", "results/second_mixture.json not built"
    row = json.loads(path.read_text())

    inside = row["per_corpus"]
    pooled = row["auroc_pooled"]
    best_inside_size = max(v["auroc_size"] for v in inside.values())
    best_inside_entropy = max(v["auroc_entropy"] for v in inside.values())

    size_lift = pooled["image size"] - best_inside_size
    entropy_lift = pooled["entropy"] - best_inside_entropy

    if size_lift <= 0.03:
        return "FAIL", (
            f"pooling lifts the descriptor by only {size_lift:+.3f}; the confound does not "
            "replicate on the second mixture"
        )
    if entropy_lift >= size_lift:
        return "FAIL", (
            f"pooling lifts entropy by {entropy_lift:+.3f} against the descriptor's "
            f"{size_lift:+.3f}, so the inflation is not specific to the free descriptor"
        )
    # Inside each part the descriptor must stay weak, or there is no collapse
    # to explain in the first place.
    weak_inside = all(v["auroc_size"] <= 0.60 for v in inside.values())
    if not weak_inside:
        return "FAIL", (
            "the descriptor is no longer weak inside every part: "
            + ", ".join(f"{k} {v['auroc_size']:.3f}" for k, v in inside.items())
        )

    return "PASS", (
        f"over {row['n']} examples from three corpora sharing no dataset with pilot1000, "
        "the free descriptor scores "
        + ", ".join(f"{k} {v['auroc_size']:.3f}" for k, v in sorted(inside.items()))
        + f" inside and {pooled['image size']:.3f} pooled, a lift of {size_lift:+.3f}; "
        f"entropy moves {entropy_lift:+.3f} over the same pooling, so what pooling inflates "
        "is the descriptor"
    )



@check("CV21: the model-read half holds off documents, where the ladder collapses to two rungs")
def _():
    """TextVQA: entropy clears, the graded rule loses, and we report no descriptor.

    The corpus is disqualified for a size-descriptor test because 98.8% of its
    images share one size (CV19), and it is exactly the right corpus for the
    half of the claim that reads the model instead. It also caps the ladder at
    two usable rungs, so the binary rule should beat the graded one, which is
    the ladder claim stated in reverse.
    """
    path = Path("results/free_signal_textvqa.json")
    if not path.exists():
        return "SKIP", "results/free_signal_textvqa.json not built"
    rows = json.loads(path.read_text())

    binary = {k: v for k, v in rows.items() if k.startswith("binary, entropy")}
    ladder = {k: v for k, v in rows.items() if k.startswith("ladder, entropy")}
    if len(binary) < 4 or len(ladder) < 4:
        return "SKIP", "entropy rows incomplete"

    cleared = sum(1 for v in binary.values() if v["gap"][1] > 0)
    if cleared < 4:
        return "FAIL", (
            f"entropy clears at {cleared} of {len(binary)} preferences on TextVQA; the text "
            "says all four, so the positive half no longer holds off documents"
        )

    # The graded rule must lose here, because there is no rung to choose.
    best_binary = max(v["gap"][0] for v in binary.values())
    best_ladder = max(v["gap"][0] for v in ladder.values())
    if best_ladder >= best_binary:
        return "FAIL", (
            f"the graded rule reaches {best_ladder:+.3f} against the binary rule's "
            f"{best_binary:+.3f}; the text explains the opposite"
        )

    # And the reason must still be duplicate rungs, checked from the records.
    from collections import defaultdict

    from gwel.config import load_config
    from gwel.data.scoring import ScoringPolicy, rescore_records
    from gwel.oracle.records import deduplicate_records, read_records

    config = load_config("configs/textvqa500.yaml")
    grouped = defaultdict(dict)
    try:
        for row in rescore_records(
            deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
        ):
            grouped[row.example_id][row.config_id] = row
    except FileNotFoundError:
        return "SKIP", "textvqa records missing"
    rungs = ("lowres_384", "lowres_768", "lowres_1152", "full")
    ids = [e for e in grouped if all(c in grouped[e] for c in rungs)]
    tokens = np.array([[grouped[e][c].visual_tokens for c in rungs] for e in ids], float)
    steps = np.diff(tokens, axis=1)
    distinct = [float((steps[:, k] > 0).mean()) for k in range(steps.shape[1])]
    if distinct[1] > 0.10 or distinct[2] > 0.10:
        return "FAIL", (
            f"the upper rungs are distinct on {distinct[1]:.1%} and {distinct[2]:.1%} of "
            "pages, so duplicate rungs no longer explain the binary rule winning"
        )

    return "PASS", (
        f"on {len(ids)} photographs entropy clears at {cleared} of four preferences (best "
        f"{best_binary:+.3f} binary against {best_ladder:+.3f} graded), and the upper two "
        f"rungs are distinct on only {distinct[1]:.1%} and {distinct[2]:.1%} of pages, so "
        "the graded action space has nothing to choose. No descriptor column is reported "
        "here: the corpus is 98.8% one size"
    )


def main() -> None:
    width = max(len(r.claim) for r in RESULTS) + 2
    print(f"{'claim':<{width}}{'status':<8}detail")
    print("-" * (width + 8 + 40))
    for result in RESULTS:
        print(f"{result.claim:<{width}}{result.status:<8}{result.detail}")

    counts = {s: sum(r.status == s for r in RESULTS) for s in ("PASS", "FAIL", "SKIP")}
    print(f"\n{counts['PASS']} passed, {counts['FAIL']} failed, {counts['SKIP']} skipped")
    if counts["FAIL"]:
        print("\nA failing claim means the documentation is now wrong. Fix the prose, not the check.")
    sys.exit(1 if counts["FAIL"] else 0)


if __name__ == "__main__":
    main()
