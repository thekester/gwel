"""Separate the three routing questions and test which signals answer them.

The escalation literature conditions on Q1 — is the cheap pass sufficient? —
but the decision that spends budget is Q2: given it failed, will escalating
recover the answer? Q3 asks whether any action can. These are distinct labels
over the same examples, and a signal that answers one need not answer another.

Usage: python scripts/analyze_recoverability.py --config configs/pilot200.yaml
"""

import argparse
from collections import defaultdict

import numpy as np

from gwel.config import load_config
from gwel.data.scoring import ScoringPolicy, rescore_records
from gwel.modeling.signals import ConfidenceSignals
from gwel.oracle.records import RunRecord, deduplicate_records, read_records
from gwel.router.evaluate import auroc

SIGNALS = ("mean_entropy", "max_entropy", "mean_logprob", "min_logprob", "mean_margin")

#: Signals where a higher value means lower confidence, so they are negated
#: before ranking so that "more confident" always sorts first.
INVERTED = {"mean_entropy", "max_entropy", "first_entropy"}


def signal_value(record: RunRecord, name: str) -> float:
    value = float(getattr(ConfidenceSignals.from_dict(record.signals), name))
    return -value if name in INVERTED else value


def auroc_with_ci(
    scores: list[float], labels: list[bool], *, resamples: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    """AUROC with a percentile bootstrap interval over examples."""
    point = auroc(scores, labels)
    array_s, array_y = np.asarray(scores), np.asarray(labels)
    rng = np.random.default_rng(seed)
    draws: list[float] = []
    for _ in range(resamples):
        index = rng.integers(0, len(scores), len(scores))
        if len(set(array_y[index].tolist())) < 2:
            continue
        draws.append(auroc(array_s[index].tolist(), array_y[index].tolist()))
    if not draws:
        return point, float("nan"), float("nan")
    low, high = np.quantile(draws, [0.025, 0.975])
    return point, float(low), float(high)


def report(name: str, question: str, scores_by_signal, labels) -> None:
    if len(set(labels)) < 2:
        print(f"\n== {name}: {question} ==\n  skipped: one class only")
        return
    base = sum(labels) / len(labels)
    print(f"\n== {name}: {question} ==")
    print(f"  n={len(labels)}, positive rate={base:.0%}")
    print(f"  {'signal':<16}{'AUROC':>8}{'95% CI':>20}")
    for signal, scores in scores_by_signal.items():
        point, low, high = auroc_with_ci(scores, labels)
        verdict = ""
        if not np.isnan(low):
            if low > 0.5:
                verdict = "  predictive"
            elif high < 0.5:
                verdict = "  inverted"
            else:
                verdict = "  uninformative"
        print(f"  {signal:<16}{point:>8.3f}   [{low:.3f}, {high:.3f}]{verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot200.yaml")
    parser.add_argument("--escalate-to", default="full", help="config id treated as the escalation")
    args = parser.parse_args()

    config = load_config(args.config)
    records = rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    )
    grouped: dict[str, dict[str, RunRecord]] = defaultdict(dict)
    for record in records:
        grouped[record.example_id][record.config_id] = record

    probe_id = config.router.feature_config_id
    usable = [
        example_id
        for example_id, configs in grouped.items()
        if probe_id in configs
        and args.escalate_to in configs
        and configs[probe_id].signals is not None
    ]
    if not usable:
        print(f"no examples carry both {probe_id} and {args.escalate_to}")
        return

    def scores_for(example_ids: list[str]) -> dict[str, list[float]]:
        return {
            signal: [signal_value(grouped[e][probe_id], signal) for e in example_ids]
            for signal in SIGNALS
        }

    report(
        "Q1",
        f"is the cheap pass ({probe_id}) correct?",
        scores_for(usable),
        [grouped[e][probe_id].correct for e in usable],
    )

    failed = [e for e in usable if not grouped[e][probe_id].correct]
    report(
        "Q2",
        f"given the cheap pass failed, does {args.escalate_to} recover it?",
        scores_for(failed),
        [grouped[e][args.escalate_to].correct for e in failed],
    )

    def answerable(example_id: str) -> bool:
        return any(
            record.correct
            for config_id, record in grouped[example_id].items()
            if config_id.startswith(("lowres_", "crop_", "ocr_"))
        )

    report(
        "Q3",
        "is this answerable by any routable action?",
        scores_for(usable),
        [answerable(e) for e in usable],
    )

    print("\n== escalation outcome breakdown ==")
    counts = {"both right": 0, "escalation helps": 0, "both wrong": 0, "escalation harms": 0}
    for example_id in usable:
        cheap = grouped[example_id][probe_id].correct
        escalated = grouped[example_id][args.escalate_to].correct
        if cheap and escalated:
            counts["both right"] += 1
        elif not cheap and escalated:
            counts["escalation helps"] += 1
        elif not cheap and not escalated:
            counts["both wrong"] += 1
        else:
            counts["escalation harms"] += 1
    total = len(usable)
    for label, count in counts.items():
        print(f"  {label:<20}{count:>5}  ({count / total:>4.0%})")
    useful = counts["escalation helps"] / total
    print(f"\n  escalation pays on {useful:.0%} of queries and harms {counts['escalation harms'] / total:.0%}")


if __name__ == "__main__":
    main()
