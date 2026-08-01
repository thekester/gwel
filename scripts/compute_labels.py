"""Derive oracle labels (cheapest correct action) from run records.

Usage: python scripts/compute_labels.py --config configs/default.yaml
"""

import argparse
from collections import Counter

from gwel.config import load_config
from gwel.oracle.cost import CostWeights
from gwel.oracle.label import derive_labels, write_labels
from gwel.oracle.records import read_records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--records", default=None, help="override the records input path")
    parser.add_argument("--out", default=None, help="override the labels output path")
    args = parser.parse_args()

    config = load_config(args.config)
    records = read_records(args.records or config.paths.records)
    weights = CostWeights.from_config(config.cost)
    labels = derive_labels(records, weights=weights)
    out_path = args.out or config.paths.labels
    write_labels(out_path, labels)

    actions = Counter(
        label.action.value if label.action is not None else "unsolvable" for label in labels
    )
    print(f"labeled {len(labels)} examples -> {out_path}")
    for name, count in sorted(actions.items()):
        print(f"  {name}: {count} ({count / len(labels):.1%})")


if __name__ == "__main__":
    main()
