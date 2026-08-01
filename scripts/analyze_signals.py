"""Test the premise: do the model's own confidence signals predict correctness?

Budget-aware routing conditioned on internal confidence only works if a cheap
pass knows when it is wrong. This script measures, per signal, the AUROC for
predicting whether a pass answered correctly, plus the risk-coverage curve of
the best signal. An AUROC near 0.5 everywhere would falsify the premise.

Usage: python scripts/analyze_signals.py --config configs/pilot20.yaml
"""

import argparse
from collections import defaultdict

from gwel.config import load_config
from gwel.data.scoring import ScoringPolicy, rescore_records
from gwel.modeling.signals import ConfidenceSignals
from gwel.oracle.records import deduplicate_records, read_records
from gwel.router.evaluate import auroc, risk_coverage

#: Signals where a *higher* value means *lower* confidence, so the score must
#: be negated before it can rank correct answers first.
INVERTED = {"mean_entropy", "max_entropy", "first_entropy"}

SIGNAL_NAMES = (
    "mean_logprob",
    "min_logprob",
    "mean_entropy",
    "max_entropy",
    "first_entropy",
    "mean_margin",
    "min_margin",
)


def signal_table(records, title: str) -> tuple[str, float] | None:
    """Print AUROC per signal; return the best (name, auroc) pair."""
    usable = [r for r in records if r.signals is not None]
    labels = [r.correct for r in usable]
    if len(set(labels)) < 2:
        print(f"\n== {title} ==\n  skipped: only one outcome class present")
        return None

    print(f"\n== {title} (n={len(usable)}, accuracy={sum(labels) / len(labels):.2f}) ==")
    print(f"  {'signal':<16}{'AUROC':>8}")
    best: tuple[str, float] | None = None
    for name in SIGNAL_NAMES:
        scores = [float(getattr(ConfidenceSignals.from_dict(r.signals), name)) for r in usable]
        if name in INVERTED:
            scores = [-s for s in scores]
        area = auroc(scores, labels)
        marker = ""
        if best is None or abs(area - 0.5) > abs(best[1] - 0.5):
            best = (name, area)
            marker = ""
        print(f"  {name:<16}{area:>8.3f}{marker}")
    if best is not None:
        print(f"  best: {best[0]} (AUROC {best[1]:.3f})")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot20.yaml")
    parser.add_argument("--records", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    records = rescore_records(
        deduplicate_records(read_records(args.records or config.paths.records)),
        ScoringPolicy(),
    )

    print(f"{len(records)} records over {len({r.example_id for r in records})} examples")
    print("AUROC > 0.5 means the signal ranks correct answers above wrong ones.")

    # Pooled over every configuration: does confidence track correctness at all?
    best = signal_table([r for r in records if r.action is not None], "all routable passes")

    # The decision the router actually faces: escalate after the cheap pass?
    feature_id = config.router.feature_config_id
    cheap = [r for r in records if r.config_id == feature_id]
    per_config_best = signal_table(cheap, f"cheap pass only ({feature_id})")

    chosen = per_config_best or best
    if chosen is not None and cheap:
        name, _ = chosen
        usable = [r for r in cheap if r.signals is not None]
        if len({r.correct for r in usable}) == 2:
            scores = [
                float(getattr(ConfidenceSignals.from_dict(r.signals), name)) for r in usable
            ]
            if name in INVERTED:
                scores = [-s for s in scores]
            curve = risk_coverage(scores, [r.correct for r in usable])
            print(f"\n== risk-coverage of {name} on {feature_id} ==")
            print(f"  AURC: {curve.aurc:.3f} (lower is better)")
            for target in (0.25, 0.5, 0.75, 1.0):
                index = min(
                    range(len(curve.coverages)),
                    key=lambda i: abs(curve.coverages[i] - target),
                )
                print(
                    f"  coverage {curve.coverages[index]:.2f} -> risk {curve.risks[index]:.2f}"
                )

    print("\n== accuracy by dataset, cheap pass vs full ==")
    by_dataset: dict[str, list] = defaultdict(list)
    for record in records:
        by_dataset[record.dataset].append(record)
    for dataset, group in sorted(by_dataset.items()):
        cheap_group = [r for r in group if r.config_id == feature_id]
        full_group = [r for r in group if r.config_id == "full"]
        if cheap_group and full_group:
            cheap_acc = sum(r.correct for r in cheap_group) / len(cheap_group)
            full_acc = sum(r.correct for r in full_group) / len(full_group)
            print(f"  {dataset:<10} cheap={cheap_acc:.2f} full={full_acc:.2f} gap={full_acc - cheap_acc:+.2f}")


if __name__ == "__main__":
    main()
