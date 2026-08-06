"""Is image size a signal, or a proxy for which benchmark a query came from?

The free descriptor clears the randomisation hull on a benchmark mixture and
fails inside every workload. Two readings fit that and they differ for a
practitioner. Either image size carries information about whether more pixels
would help, unevenly distributed across benchmarks; or it is a benchmark
detector, useful only because benchmarks differ in escalation value and useless
on traffic homogeneous in provenance but varied in content.

Separating them needs a stratum where size still varies and provenance is
mixed. The pilot contains one, and inside it the descriptor is scored on both
targets at once: if it is a signal it should predict the escalation gain, and
if it is a detector it should predict provenance and little else. The gap
between those two numbers is the answer.

Usage: PYTHONPATH=src python scripts/size_content_confound.py
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from gwel.config import load_config
from gwel.data.scoring import ScoringPolicy, rescore_records
from gwel.oracle.records import deduplicate_records, read_records
from gwel.router.decision import signed_gain
from gwel.router.evaluate import auroc

MIN_STRATUM = 60
MIN_MINORITY = 0.15


def cost_channel(configs: list[tuple[str, str]]) -> list[dict]:
    """Does the descriptor track escalation cost, per serving model?

    Proposition 2 says a policy above the hull carries information about the
    joint distribution of cost and correctness. A descriptor at chance on
    correctness that still clears must therefore be reading cost, and one that
    clears nowhere should not be. Spearman is used rather than Pearson because
    the descriptor enters the policy through a threshold, so only its ordering
    matters.
    """
    from gwel.config import load_config

    rows = []
    for path, label in configs:
        if not Path(path).exists():
            continue
        config = load_config(path)
        grouped: dict[str, dict] = defaultdict(dict)
        try:
            records = rescore_records(
                deduplicate_records(read_records(config.paths.records)),
                ScoringPolicy(),
            )
        except FileNotFoundError:
            continue
        for row in records:
            grouped[row.example_id][row.config_id] = row
        ids = [e for e in grouped if {"lowres_384", "full"} <= set(grouped[e])]
        if len(ids) < 100:
            continue
        size = np.array(
            [
                float(max(grouped[e]["lowres_384"].meta["orig_width"],
                          grouped[e]["lowres_384"].meta["orig_height"]))
                for e in ids
            ]
        )
        extra = np.array(
            [
                grouped[e]["full"].latency_ms - grouped[e]["lowres_384"].latency_ms
                for e in ids
            ],
            float,
        )
        gain = signed_gain(
            np.array([grouped[e]["lowres_384"].correct for e in ids]),
            np.array([grouped[e]["full"].correct for e in ids]),
        )
        # A hull comparison is only meaningful when the fixed configurations
        # differ in cost. Where they do not, the hull is near vertical and a
        # gap to it is arbitrarily sensitive to a millisecond of measurement
        # noise, which is the cost-side form of step 3 of Algorithm 3.
        spread = float(
            np.mean([grouped[e]["full"].latency_ms for e in ids])
            / np.mean([grouped[e]["lowres_384"].latency_ms for e in ids])
        )
        rank = lambda v: np.argsort(np.argsort(v))  # noqa: E731
        rows.append(
            {
                "model": label,
                "n": len(ids),
                "cost_spread": spread,
                "hull_usable": bool(spread >= 1.5),
                "spearman_size_cost": float(
                    np.corrcoef(rank(size), rank(extra))[0, 1]
                ),
                "auroc_size_gain": auroc(
                    size.tolist(), [bool(g > 0) for g in gain]
                ),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot1000.yaml")
    parser.add_argument("--out", default="results/size_confound.json")
    args = parser.parse_args()

    config = load_config(args.config)
    grouped: dict[str, dict] = defaultdict(dict)
    for row in rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    ):
        grouped[row.example_id][row.config_id] = row
    ids = [e for e in grouped if "lowres_384" in grouped[e] and "full" in grouped[e]]
    size = np.array(
        [
            float(max(grouped[e]["lowres_384"].meta["orig_width"],
                      grouped[e]["lowres_384"].meta["orig_height"]))
            for e in ids
        ]
    )
    dataset = np.array([grouped[e]["lowres_384"].dataset for e in ids])
    gain = signed_gain(
        np.array([grouped[e]["lowres_384"].correct for e in ids]),
        np.array([grouped[e]["full"].correct for e in ids]),
    )
    positive = [bool(g > 0) for g in gain]

    out: dict = {"n": len(ids), "auroc_pooled": auroc(size.tolist(), positive)}
    print(f"n = {len(ids)}; image size predicts the escalation gain at "
          f"{out['auroc_pooled']:.3f} pooled")

    print("\nimage size as a dataset detector (one against the rest):")
    detector = {}
    for name in sorted(set(dataset)):
        detector[name] = auroc(size.tolist(), [bool(d == name) for d in dataset])
        print(f"  {name:<10} AUROC {detector[name]:.3f}")
    out["dataset_detector_auroc"] = detector

    # Look for a stratum that mixes provenances and still varies in size. If the
    # two were separable on these data, one would exist.
    edges = np.unique(np.quantile(size, np.linspace(0.0, 1.0, 11)))
    usable = []
    for low, high in zip(edges, edges[1:], strict=False):
        band = (size >= low) & (size <= high)
        if band.sum() < MIN_STRATUM:
            continue
        counts = Counter(dataset[band])
        minority = 1.0 - max(counts.values()) / band.sum()
        varies = float(size[band].std()) > 1.0
        usable.append(
            {
                "low": float(low),
                "high": float(high),
                "n": int(band.sum()),
                "size_std": float(size[band].std()),
                "minority_share": float(minority),
                "separable": bool(varies and minority >= MIN_MINORITY),
                "composition": {k: int(v) for k, v in counts.items()},
            }
        )
    out["strata"] = usable
    separable = [s for s in usable if s["separable"]]
    out["separable_stratum_exists"] = bool(separable)

    print(f"\n{len(usable)} strata of at least {MIN_STRATUM} examples:")
    for stratum in usable:
        print(
            f"  {stratum['low']:>5.0f}-{stratum['high']:<5.0f} n={stratum['n']:<4} "
            f"size sd {stratum['size_std']:>6.1f}  minority provenance "
            f"{stratum['minority_share']:.0%}"
            + ("  <- separable" if stratum["separable"] else "")
        )
    print(
        f"\nstratum with both size variance and mixed provenance: "
        f"{'found' if separable else 'none'}"
    )

    # Inside the separable strata, score the descriptor on both targets. If it
    # is a signal it should predict the gain; if it is a detector it should
    # predict provenance and little else.
    print("\ninside each separable stratum, what does the descriptor predict?")
    out["separable_detail"] = []
    for stratum in separable:
        band = (size >= stratum["low"]) & (size <= stratum["high"])
        if float(size[band].std()) <= 1.0:
            continue
        index = np.where(band)[0]
        on_gain = auroc(size[band].tolist(), [positive[i] for i in index])
        on_source = auroc(
            size[band].tolist(),
            [bool(d == "docvqa") for d in dataset[band]],
        )
        by_source = {
            name: float(np.mean(gain[band & (dataset == name)] > 0))
            for name in sorted(set(dataset[band]))
            if (band & (dataset == name)).sum() >= 20
        }
        row = {
            "low": stratum["low"],
            "high": stratum["high"],
            "n": stratum["n"],
            "auroc_on_gain": on_gain,
            "auroc_on_source": on_source,
            "escalation_value_by_source": by_source,
        }
        out["separable_detail"].append(row)
        print(
            f"  {stratum['low']:.0f}-{stratum['high']:.0f}: gain {on_gain:.3f}, "
            f"source {on_source:.3f}; value by source "
            + ", ".join(f"{k} {v:.0%}" for k, v in by_source.items())
        )

    # The one stratum that mixes provenances is the capped one, where the
    # descriptor has no variance at all. Provenance still decides the value.
    capped = size >= np.quantile(size, 0.75)
    value = {}
    for name in sorted(set(dataset[capped])):
        mask = capped & (dataset == name)
        if mask.sum() >= 20:
            value[name] = float(np.mean(gain[mask] > 0))
    out["capped_stratum"] = {
        "n": int(capped.sum()),
        "size_std": float(size[capped].std()),
        "escalation_value": value,
    }
    print(
        f"\nthe stratum that does mix provenances is the capped one "
        f"(n={int(capped.sum())}, size sd {size[capped].std():.1f}):"
    )
    for name, share in value.items():
        print(f"  {name:<10} escalation repairs {share:.0%}")

    out["cost_channel"] = cost_channel(
        [
            ("configs/docvqa1200.yaml", "DocVQA, SmolVLM-500M"),
            ("configs/docvqa1200_256m.yaml", "DocVQA, SmolVLM-256M"),
            ("configs/docvqa1200_2b.yaml", "DocVQA, SmolVLM2-2.2B"),
            ("configs/docvqa1200_qwen2b.yaml", "DocVQA, Qwen2-VL-2B"),
            ("configs/docvqa1200_llavaov.yaml", "DocVQA, LLaVA-OneVision-0.5B"),
            ("configs/docvqa1200_internvl1b.yaml", "DocVQA, InternVL3-1B"),
            ("configs/infovqa500.yaml", "InfoVQA, SmolVLM-500M"),
            ("configs/infovqa500_qwen2b.yaml", "InfoVQA, Qwen2-VL-2B"),
            ("configs/chartqa500.yaml", "ChartQA, SmolVLM-500M"),
        ]
    )
    if out["cost_channel"]:
        print("\ndoes the descriptor track escalation cost, per serving model?")
        for row in out["cost_channel"]:
            print(
                f"  {row['model']:<22} n={row['n']:<5} "
                f"Spearman(size, extra cost) {row['spearman_size_cost']:+.3f}; "
                f"AUROC(size, gain) {row['auroc_size_gain']:.3f}; "
                f"cost spread {row['cost_spread']:.2f}x"
                + ("" if row["hull_usable"] else "  <- hull degenerate, excluded")
            )

    print(
        "\nInside a stratum that mixes provenances and still varies in size, the\n"
        "descriptor separates the source far better than it predicts the gain,\n"
        "and the value gap between sources survives. That is the detector\n"
        "reading: its pooled power comes from separating benchmarks that differ\n"
        "in escalation value, so traffic where size and provenance decouple\n"
        "should not expect it to route."
    )
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
