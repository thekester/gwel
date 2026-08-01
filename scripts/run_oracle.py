"""Run every visual configuration on the pilot and log instrumented records.

The run is resumable: records already present in the output file are skipped.

Usage: python scripts/run_oracle.py --config configs/default.yaml [--limit N]
"""

import argparse
import logging

from gwel.config import load_config
from gwel.data.loaders import read_manifest
from gwel.oracle.runner import OracleRunner


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--limit", type=int, default=None, help="cap the number of examples")
    parser.add_argument("--out", default=None, help="override the records output path")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    # NVML initialization before CUDA Torch can break c10.dll loading on
    # Windows. The inference CLI requires Torch, so establish the safe order.
    import torch  # noqa: F401

    config = load_config(args.config)
    examples = read_manifest(config.paths.pilot_manifest)
    if args.limit is not None:
        examples = examples[: args.limit]

    runner = OracleRunner(config)
    out_path = args.out or config.paths.records
    print(f"energy backends: {runner.energy_meter.backend_names or '(none)'}")
    counters = runner.run(examples, out_path)
    print(f"records -> {out_path}")
    for name, value in counters.items():
        print(f"  {name}: {value}")
    if runner.engine.load_report is not None:
        report = runner.engine.load_report
        print(f"model load: {report.load_ms:.0f} ms, +{report.ram_delta_mb:.0f} MB RSS")


if __name__ == "__main__":
    main()
