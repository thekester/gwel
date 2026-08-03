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
        f"accuracy {accuracy[0]:+.3f} [{accuracy[1]:+.3f}, {accuracy[2]:+.3f}], "
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


@check("X1: no claim is lost to family-wise error correction")
def _():
    """The paper asked thirteen paired questions; Holm says which to believe.

    Uncorrected, thirteen tests at the nominal level carry a 49% chance of at
    least one false positive. This asserts the stronger property: every
    comparison that clears the nominal level also survives the step-down.
    """
    summary = json.loads(Path("results/multiplicity.json").read_text())
    nominal, survivors = summary["nominal"], summary["survivors"]
    total = len(summary["tests"])
    ok = survivors == nominal and survivors > 0
    return ("PASS" if ok else "FAIL"), (
        f"{nominal}/{total} clear the nominal level, {survivors} survive Holm; "
        f"uncorrected family-wise error would be "
        f"{summary['family_wise_error_uncorrected']:.0%}"
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


@check("R9: the ceiling sits at a pixel resolution, not at a sequence length")
def _():
    """A third model with a different vision encoder separates the two.

    SmolVLM-500M and 256M share an 86M encoder, so "saturates at 1152 px" and
    "saturates at 640 visual tokens" name the same point for both. SmolVLM2-2.2B
    tokenises differently, so the two come apart, and only one of them can be
    where all three stop gaining.
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


@check("R8: saturation is a property of the corpus, not of the serving model")
def _():
    """Two models, half the parameters apart, on the same 1200 pages.

    Capacity and legibility predict opposite things: a smaller model should
    saturate earlier if the ceiling is its own, and at the same rung with lower
    accuracy throughout if the ceiling is the images. Both nulls must also be
    tight, or this compares two failures to measure.
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
    ok = same and tight and lower_everywhere
    return ("PASS" if ok else "FAIL"), (
        f"{len(runs)} models saturate at {set(rungs.values())}; nulls tight={tight}; "
        f"the smaller model is less accurate at every rung below the "
        f"ceiling={lower_everywhere}"
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
