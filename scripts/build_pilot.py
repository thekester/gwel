"""Build the pilot dataset manifest from streamed HuggingFace sources.

Usage: python scripts/build_pilot.py --config configs/default.yaml
"""

import argparse
import logging
from collections import Counter

from gwel.config import load_config
from gwel.data.loaders import build_pilot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    config = load_config(args.config)
    examples = build_pilot(config.datasets, manifest_path=config.paths.pilot_manifest)
    counts = Counter(example.dataset for example in examples)
    print(f"wrote {len(examples)} examples to {config.paths.pilot_manifest}")
    for name, count in sorted(counts.items()):
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
