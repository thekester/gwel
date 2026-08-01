"""Show where each pipeline stage stands for a given config.

Safe to run at any time, including while a run is in progress: it only reads
files. Use it to watch a long oracle run without guessing at record counts.

Usage: python scripts/status.py [--config configs/pilot200.yaml] [--watch]
"""

import argparse
import json
import time
from pathlib import Path

from gwel.config import load_config


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _bar(fraction: float, width: int = 28) -> str:
    filled = int(round(min(max(fraction, 0.0), 1.0) * width))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def _age(path: Path) -> str:
    if not path.exists():
        return ""
    seconds = time.time() - path.stat().st_mtime
    if seconds < 90:
        return f"{seconds:.0f}s ago"
    if seconds < 5400:
        return f"{seconds / 60:.0f}m ago"
    return f"{seconds / 3600:.1f}h ago"


def report(config_path: str) -> bool:
    """Print one status snapshot; return True when the oracle run is complete."""
    from gwel.oracle.runner import planned_config_ids

    config = load_config(config_path)
    manifest = Path(config.paths.pilot_manifest)
    records = Path(config.paths.records)
    labels = Path(config.paths.labels)
    checkpoint = Path(config.paths.router_dir) / "router.json"

    planned = len(planned_config_ids(config))
    n_examples = _count_lines(manifest)
    n_records = _count_lines(records)
    expected = n_examples * planned
    complete = expected > 0 and n_records >= expected

    print(f"config: {config_path}")
    print(f"  1. pilot     {n_examples:>6} examples   {_age(manifest)}")

    if expected:
        fraction = n_records / expected
        print(
            f"  2. oracle    {n_records:>6}/{expected} records "
            f"{_bar(fraction)} {fraction:>5.0%}  {_age(records)}"
        )
        if 0 < n_records < expected:
            done_examples = n_records // planned
            print(f"               ~{done_examples}/{n_examples} examples done")
    else:
        print(f"  2. oracle    {n_records:>6} records (build the pilot first)")

    n_labels = _count_lines(labels)
    if n_labels:
        actions: dict[str, int] = {}
        with labels.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    action = json.loads(line)["action"] or "unsolvable"
                    actions[action] = actions.get(action, 0) + 1
        mix = "  ".join(f"{k}={v}" for k, v in sorted(actions.items()))
        print(f"  3. labels    {n_labels:>6} examples   {_age(labels)}")
        print(f"               {mix}")
    else:
        print(f"  3. labels         - not computed")

    if checkpoint.exists():
        print(f"  4. router    trained            {_age(checkpoint)}")
    else:
        print(f"  4. router         - not trained")

    if complete:
        print("\n  oracle run complete -> compute_labels, train_router, evaluate_router")
    return complete


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/pilot200.yaml")
    parser.add_argument("--watch", action="store_true", help="refresh until the run completes")
    parser.add_argument("--interval", type=float, default=30.0)
    args = parser.parse_args()

    while True:
        done = report(args.config)
        if not args.watch or done:
            break
        print(f"\n(refreshing every {args.interval:.0f}s, Ctrl-C to stop)\n")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
