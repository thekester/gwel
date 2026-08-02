"""Decide whether a run's energy measurements are trustworthy enough to report.

Two checks, both falsifiable:

1. **Agreement.** Configurations spending the same number of visual tokens
   should cost the same energy. Their disagreement is a direct estimate of the
   instrument's noise floor, with no modelling assumptions.
2. **Resolution.** That noise floor must be small compared to the effect being
   claimed, here, the energy difference between the cheapest and the most
   expensive configuration.

A run failing these should not be used for any energy claim, however large the
apparent effect.

Usage: python scripts/validate_energy.py --config configs/pilot200.yaml
"""

import argparse
from collections import defaultdict

import numpy as np

from gwel.config import load_config
from gwel.oracle.records import RunRecord, deduplicate_records, read_records

#: Disagreement between equal-token configurations above which energy numbers
#: are not reportable. Ten percent is already generous for a hardware claim.
AGREEMENT_THRESHOLD = 0.10

#: Minimum ratio of claimed effect to noise floor for the effect to be real.
SIGNAL_TO_NOISE_THRESHOLD = 3.0


def group_by_tokens(records: list[RunRecord]) -> dict[int, dict[str, list[float]]]:
    """Median energies per config, bucketed by visual token count."""
    grouped: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for record in records:
        if record.net_energy_mj is not None:
            grouped[record.visual_tokens][record.config_id].append(record.net_energy_mj)
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot200.yaml")
    parser.add_argument("--records", default=None)
    parser.add_argument(
        "--min-samples", type=int, default=10, help="configs with fewer records are ignored"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    records = deduplicate_records(read_records(args.records or config.paths.records))
    # OCR runs a CPU tool inside the GPU measurement window, so its energy is
    # not attributable to the same component and cannot join this comparison.
    records = [r for r in records if not r.config_id.startswith("ocr_")]
    if not records:
        print("no GPU-only records with energy; nothing to validate")
        return

    print(f"{len(records)} records from {args.records or config.paths.records}\n")

    repeats = records[0].meta.get("repeats", 1)
    print(f"runner.repeats = {repeats}" + ("  (single-shot measurements)" if repeats == 1 else ""))

    print("\n== dispersion within each configuration ==")
    print(f"  {'config':<14}{'n':>5}{'median mJ':>12}{'IQR/median':>12}")
    per_config: dict[str, float] = {}
    for config_id in sorted({r.config_id for r in records}):
        values = np.array(
            [r.net_energy_mj for r in records if r.config_id == config_id and r.net_energy_mj]
        )
        if len(values) < args.min_samples:
            continue
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        per_config[config_id] = float(median)
        print(f"  {config_id:<14}{len(values):>5}{median:>12.0f}{(q3 - q1) / median:>12.0%}")

    print("\n== agreement between equal-token configurations ==")
    worst = 0.0
    compared = 0
    for tokens, configs in sorted(group_by_tokens(records).items()):
        medians = {
            config_id: float(np.median(values))
            for config_id, values in configs.items()
            if len(values) >= args.min_samples
        }
        if len(medians) < 2:
            continue
        low, high = min(medians.values()), max(medians.values())
        spread = (high - low) / low
        worst = max(worst, spread)
        compared += 1
        status = "ok" if spread <= AGREEMENT_THRESHOLD else "FAIL"
        detail = "  ".join(f"{c}={m:.0f}" for c, m in sorted(medians.items()))
        print(f"  {tokens:>4} tokens: {detail}")
        print(f"              spread {spread:>6.0%}  [{status}]")

    if compared == 0:
        print("  no token count is covered by two configurations; cannot estimate noise")
        print("\nVERDICT: unvalidated: add configurations that share a token count")
        return

    effect = (max(per_config.values()) - min(per_config.values())) / min(per_config.values())
    ratio = effect / worst if worst > 0 else float("inf")
    print(f"\n== resolution ==")
    print(f"  largest effect across configs: {effect:>6.0%}")
    print(f"  noise floor (worst spread):    {worst:>6.0%}")
    print(f"  signal-to-noise:               {ratio:>6.1f}x")

    passed = worst <= AGREEMENT_THRESHOLD and ratio >= SIGNAL_TO_NOISE_THRESHOLD
    print(
        f"\nVERDICT: {'energy measurements are reportable' if passed else 'DO NOT report energy'}"
    )
    if not passed:
        print(
            "  raise runner.repeats, lengthen the measurement window, and re-check;\n"
            "  latency, token counts and accuracy are unaffected by this failure."
        )


if __name__ == "__main__":
    main()
