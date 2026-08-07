"""A second, disjoint mixture, to answer the objection that we have only one.

The two-regime claim compares one benchmark mixture against single workloads,
and eight of our corpus-model pairs sit on DocVQA. A reviewer is entitled to
read the whole result as pilot1000-versus-DocVQA. The cleanest answer costs no
new inference: our three single-domain corpora, all collected on the same
serving model with the same ladder, can be pooled into a mixture that shares no
dataset and no scale with pilot1000.

It is also a harder test. In pilot1000 the four benchmarks arrive at three
distinct modal sizes (640, 1024, 2048), so a size descriptor is close to a
dataset label. Here two of the three corpora are modal at 2048 px and the third
at 800, so the label is degraded on purpose. If the descriptor still separates
across this mixture and still collapses inside each part, the regime effect is
not a property of pilot1000's packaging.

Reported per corpus and pooled: the descriptor's AUROC on the escalation
target, and the hull gap of both ladders under per-example prices.

Usage: PYTHONPATH=src:scripts python scripts/second_mixture.py
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
from gwel.router.decision import fit_ladder_rule, signed_gain
from gwel.router.evaluate import bootstrap_interval

from baseline_convexity import hull_accuracy
from free_signal_single_domain import auroc

CHEAP = "lowres_384"
RUNGS = ("lowres_768", "lowres_1152", "full")
LADDER = (CHEAP, *RUNGS)
CORPORA = (
    ("configs/docvqa1200.yaml", "DocVQA"),
    ("configs/infovqa500.yaml", "InfographicVQA"),
    ("configs/chartqa500.yaml", "ChartQA"),
)
VALUES = (400.0, 800.0, 1600.0, 3200.0)
RESAMPLES = 30
HELDOUT = 600


def load(config_path: str) -> tuple[list[str], dict]:
    config = load_config(config_path)
    grouped: dict[str, dict] = defaultdict(dict)
    for row in rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    ):
        grouped[row.example_id][row.config_id] = row
    ids = [e for e in grouped if all(c in grouped[e] for c in LADDER)]
    return ids, grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/second_mixture.json")
    args = parser.parse_args()

    pooled_ids: list[tuple[str, str, dict]] = []
    per_corpus: dict[str, dict] = {}
    for path, label in CORPORA:
        ids, grouped = load(path)
        pooled_ids.extend((label, e, grouped) for e in ids)
        per_corpus[label] = {"n": len(ids)}

    def column(getter):
        return np.array([getter(g[e]) for _, e, g in pooled_ids], float)

    ok = {c: column(lambda row, c=c: row[c].correct) for c in LADDER}
    tok = {c: column(lambda row, c=c: row[c].visual_tokens) for c in LADDER}
    size = column(
        lambda row: max(
            row[CHEAP].meta.get("orig_width", 0), row[CHEAP].meta.get("orig_height", 0)
        )
    )
    entropy = column(lambda row: row[CHEAP].signals.get("mean_entropy", 0.0))
    corpus = np.array([label for label, _, _ in pooled_ids])

    # Price every pass by the affine token-cost model, as the single-domain
    # comparisons do, so the two are read on the same basis.
    buckets: dict[int, list[float]] = defaultdict(list)
    for _, e, g in pooled_ids:
        for row in g[e].values():
            buckets[int(row.visual_tokens)].append(row.latency_ms)
    usable = [t for t in sorted(buckets) if len(buckets[t]) >= 20]
    model = fit_token_cost(usable, [float(np.median(buckets[t])) for t in usable])
    ms = {c: model.predict(tok[c]) for c in LADDER}

    gains = {r: signed_gain(ok[CHEAP], ok[r]) for r in RUNGS}
    top_gain = signed_gain(ok[CHEAP], ok[RUNGS[-1]])
    deltas = np.column_stack([ms[r] - ms[CHEAP] for r in RUNGS])

    out: dict[str, object] = {"n": len(pooled_ids)}
    out["auroc_pooled"] = {
        "image size": auroc(size.tolist(), [bool(g > 0) for g in top_gain]),
        "entropy": auroc(entropy.tolist(), [bool(g > 0) for g in top_gain]),
    }
    for label in per_corpus:
        mask = corpus == label
        per_corpus[label]["auroc_size"] = auroc(
            size[mask].tolist(), [bool(g > 0) for g in top_gain[mask]]
        )
        per_corpus[label]["auroc_entropy"] = auroc(
            entropy[mask].tolist(), [bool(g > 0) for g in top_gain[mask]]
        )
    out["per_corpus"] = per_corpus

    rows: dict[str, list] = defaultdict(list)
    for seed in range(RESAMPLES):
        rng = np.random.default_rng(41000 + seed)
        order = rng.permutation(len(pooled_ids))
        test, train = order[:HELDOUT], order[HELDOUT:]
        fixed = [
            (float(ms[c][test].mean()), float(ok[c][test].mean())) for c in LADDER
        ]
        for name, signal in (("entropy", entropy), ("image size", size)):
            for value in VALUES:
                chosen = fit_ladder_rule(
                    signal[train],
                    {r: gains[r][train] for r in RUNGS},
                    value_ms_per_correct=value,
                ).choose(signal[test], deltas[test])
                accuracy = np.where(chosen < 0, ok[CHEAP][test], 0.0)
                latency = np.where(chosen < 0, ms[CHEAP][test], 0.0)
                for level, rung in enumerate(RUNGS):
                    mask = chosen == level
                    accuracy = np.where(mask, ok[rung][test], accuracy)
                    latency = np.where(mask, ms[rung][test], latency)
                mean_accuracy, mean_latency = float(accuracy.mean()), float(latency.mean())
                hull = hull_accuracy(fixed, mean_latency)
                rows[f"{name} V={value:.0f}"].append(
                    mean_accuracy - (hull if hull is not None else mean_accuracy)
                )

    out["hull_gaps"] = {}
    for name, values in rows.items():
        interval = bootstrap_interval(values)
        out["hull_gaps"][name] = {
            "gap": [interval.estimate, interval.low, interval.high],
            "gap_vector": [float(v) for v in values],
        }

    print(f"second mixture: {len(pooled_ids)} examples over three corpora, SmolVLM-500M\n")
    print(f"{'corpus':<18}{'n':>6}{'size AUROC':>13}{'entropy AUROC':>15}")
    for label, row in per_corpus.items():
        print(
            f"{label:<18}{row['n']:>6}{row['auroc_size']:>13.3f}{row['auroc_entropy']:>15.3f}"
        )
    print(
        f"{'pooled':<18}{len(pooled_ids):>6}"
        f"{out['auroc_pooled']['image size']:>13.3f}"
        f"{out['auroc_pooled']['entropy']:>15.3f}"
    )
    print(f"\n{'policy':<22}{'gap to hull [95% CI]':>28}")
    for name, row in out["hull_gaps"].items():
        estimate, low, high = row["gap"]
        print(f"{name:<22}{f'{estimate:+.3f} [{low:+.3f}, {high:+.3f}]':>28}")

    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
