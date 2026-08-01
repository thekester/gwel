"""Train the supervised router by distilling oracle labels.

Usage: python scripts/train_router.py --config configs/default.yaml
"""

import argparse
import logging

from gwel.config import load_config
from gwel.oracle.label import read_labels
from gwel.oracle.records import deduplicate_records, read_records
from gwel.router.train import build_routing_dataset, train_router


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    config = load_config(args.config)
    records = deduplicate_records(read_records(config.paths.records))
    labels = read_labels(config.paths.labels)
    dataset = build_routing_dataset(
        records, labels, feature_config_id=config.router.feature_config_id
    )
    result = train_router(dataset, config.router, out_dir=config.paths.router_dir)

    print(f"checkpoint -> {result.checkpoint_dir}")
    print(f"train acc: {result.train_accuracy:.3f} (n={result.n_train})")
    print(f"val acc:   {result.val_accuracy:.3f} (n={result.n_val})")
    print(f"test acc:  {result.test_accuracy:.3f} (n={result.n_test}, held out)")
    for action, accuracy in result.val_accuracy_per_action.items():
        print(f"  {action}: {accuracy:.3f}")


if __name__ == "__main__":
    main()
