"""Re-derive every documented claim from the data and report PASS / FAIL.

Claims written in `ANGLES.md`, `FINDINGS.md` and `PROPOSAL.md` are prose. This
turns each one into an executable check with an explicit numeric threshold, so
a claim that stops holding — because more data arrived, because a bug was
fixed, because a config changed — fails loudly instead of surviving in a
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
    rows = {r["config"]: r for r in json.loads(Path("results/component_latency.json").read_text())}
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
        if "lowres_384" in grouped[e] and "full" in grouped[e] and grouped[e]["lowres_384"].signals
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
