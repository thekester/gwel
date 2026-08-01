"""Measure cold vs warm start of the lazily loaded tools in fresh processes.

Usage: python scripts/measure_coldstart.py --config configs/default.yaml [--tools pytesseract smolvlm]
"""

import argparse
import json
from pathlib import Path

from gwel.config import load_config
from gwel.profiling.coldstart import measure_cold_start


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--tools", nargs="+", default=["pytesseract", "smolvlm"])
    parser.add_argument("--out", default="results/coldstart.jsonl")
    args = parser.parse_args()

    config = load_config(args.config)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="\n") as handle:
        for tool in args.tools:
            kwargs = {"model_id": config.model.model_id} if tool == "smolvlm" else {}
            report = measure_cold_start(tool, **kwargs)
            handle.write(json.dumps(report.to_dict(), sort_keys=True) + "\n")
            if report.error:
                print(f"{tool}: ERROR {report.error}")
            else:
                print(
                    f"{tool}: cold {report.cold_ms:.0f} ms, warm {report.warm_ms:.0f} ms, "
                    f"+{report.ram_delta_mb:.0f} MB RSS"
                )
    print(f"reports -> {out_path}")


if __name__ == "__main__":
    main()
