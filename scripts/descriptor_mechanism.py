"""What carries the free descriptor's gap, once cost allocation is subtracted.

The descriptor clears the randomisation hull on the pairs whose serving model
prices resolution steeply, while scoring 0.508 to 0.586 at predicting whether
escalation helps. A graded cost-only policy accounts for about half of that
(scripts/cost_only_graded.py) and nothing we had tried accounted for the rest.

One clue sits in the artefacts and we had not read it: the descriptor's
*binary* policies sit at +0.000 on every pair, while its *graded* ladders are
what clear. Whatever it supplies is therefore not a whether-signal at all. This
script tests the hypothesis that follows: the descriptor predicts *how far* to
escalate rather than whether to, which is a question only a graded action space
can ask and which the AUROC on the top-rung gain cannot see.

Four quantities per pair:

  whether   AUROC of image size for "does the top rung repair this query",
            the target the paper has been scoring all along;
  how far   AUROC of image size for "does this query need more than the first
            rung above cheap", restricted to the queries escalation repairs,
            which is the question a ladder faces once it has decided to spend;
  spread    AUROC of image size for "is the per-example gain dispersed across
            rungs", the variance hypothesis;
  length    AUROC of image size for "do the cheap and top answers differ in
            length", the shallow-cue hypothesis.

If the how-far column separates while the whether column does not, the
descriptor's contribution is a rung-selection signal and the paper's graded
action space is what lets it pay.

Usage: PYTHONPATH=src:scripts python scripts/descriptor_mechanism.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from gwel.config import load_config
from gwel.data.scoring import ScoringPolicy, rescore_records
from gwel.oracle.records import deduplicate_records, read_records

CHEAP = "lowres_384"

PAIRS = (
    ("configs/docvqa1200.yaml", "DocVQA, SmolVLM-500M", ("lowres_768", "lowres_1152", "full")),
    ("configs/docvqa1200_256m.yaml", "DocVQA, SmolVLM-256M", ("lowres_768", "lowres_1152", "full")),
    ("configs/docvqa1200_2b.yaml", "DocVQA, SmolVLM2-2.2B", ("lowres_768", "lowres_1152", "full")),
    ("configs/infovqa500.yaml", "InfoVQA, SmolVLM-500M", ("lowres_768", "lowres_1152", "full")),
    ("configs/infovqa500_qwen2b.yaml", "InfoVQA, Qwen2-VL-2B", ("lowres_768", "lowres_1152", "full")),
    ("configs/docvqa1200_qwen2b.yaml", "DocVQA, Qwen2-VL-2B", ("lowres_768", "lowres_1152", "full")),
    ("configs/chartqa500_llavaov.yaml", "ChartQA, LLaVA-OV-0.5B", ("lowres_768", "full")),
    ("configs/docvqa1200_llavaov.yaml", "DocVQA, LLaVA-OV-0.5B", ("lowres_768", "full")),
)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float | None:
    """Rank-based AUROC, ties averaged. None when a class is missing."""
    positive = labels > 0.5
    if positive.sum() == 0 or (~positive).sum() == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks within ties so a constant score scores exactly 0.5.
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    n_pos = float(positive.sum())
    n_neg = float((~positive).sum())
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap_auroc(scores: np.ndarray, labels: np.ndarray, seed: int = 7) -> list[float]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(200):
        pick = rng.integers(0, len(scores), len(scores))
        value = auroc(scores[pick], labels[pick])
        if value is not None:
            out.append(value)
    return out


def measure(config_path: str, rungs: tuple[str, ...]) -> dict | None:
    config = load_config(config_path)
    grouped: dict[str, dict] = defaultdict(dict)
    try:
        records = rescore_records(
            deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
        )
    except FileNotFoundError:
        return None
    for row in records:
        grouped[row.example_id][row.config_id] = row
    ladder = (CHEAP, *rungs)
    ids = [e for e in grouped if all(c in grouped[e] for c in ladder)]
    if len(ids) < 100:
        return None

    correct = np.array([[grouped[e][c].correct for c in ladder] for e in ids], float)
    size = np.array(
        [
            max(
                grouped[e][CHEAP].meta.get("orig_width", 0),
                grouped[e][CHEAP].meta.get("orig_height", 0),
            )
            for e in ids
        ],
        float,
    )
    lengths = np.array(
        [[len(str(grouped[e][c].answer or "")) for c in ladder] for e in ids], float
    )

    repaired = (correct[:, -1] > correct[:, 0]).astype(float)
    # Among repaired queries, does the first rung above cheap already suffice?
    first_rung_enough = (correct[:, 1] > correct[:, 0]).astype(float)
    needs_more = ((repaired > 0) & (first_rung_enough == 0)).astype(float)
    # Dispersion of the per-rung gain, which the mean-gain target cannot see.
    dispersed = (np.diff(correct, axis=1).std(axis=1) > 0).astype(float)
    length_changed = (np.abs(lengths[:, -1] - lengths[:, 0]) > 2).astype(float)

    out: dict[str, object] = {"n": len(ids), "repaired": float(repaired.mean())}

    def record(name: str, labels: np.ndarray, mask: np.ndarray | None = None) -> None:
        s, y = (size, labels) if mask is None else (size[mask], labels[mask])
        if len(s) < 40:
            out[name] = None
            return
        point = auroc(s, y)
        if point is None:
            out[name] = None
            return
        vector = bootstrap_auroc(s, y)
        # The percentile interval of the replicates themselves. Passing them to
        # bootstrap_interval would estimate the standard error of their mean,
        # which is smaller by the square root of their count and would report an
        # AUROC at n=59 to three decimal places.
        low, high = (float(v) for v in np.percentile(vector, [2.5, 97.5]))
        out[name] = {
            "auroc": point,
            "ci": [low, high],
            "n": int(len(s)),
            "vector": [float(v) for v in vector],
        }

    record("whether", repaired)
    record("how_far", needs_more, mask=repaired > 0)
    record("spread", dispersed)
    record("length", length_changed)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="results/descriptor_mechanism.json")
    args = parser.parse_args()

    results = {}
    header = f"{'corpus, serving model':<24}{'n':>6}{'whether':>10}{'how far':>10}{'spread':>10}{'length':>10}"
    print(header)
    for path, label, rungs in PAIRS:
        if not Path(path).exists():
            continue
        row = measure(path, rungs)
        if row is None:
            continue
        results[label] = row

        def show(key: str) -> str:
            cell = row.get(key)
            return "  n/a" if not cell else f"{cell['auroc']:.3f}"

        print(
            f"{label:<24}{row['n']:>6}{show('whether'):>10}{show('how_far'):>10}"
            f"{show('spread'):>10}{show('length'):>10}"
        )

    Path(args.out).write_text(json.dumps(results, indent=2))
    print(
        "\n'whether' is the target the paper has scored all along. 'how far' is\n"
        "the question a graded ladder faces once it has decided to spend, asked\n"
        "only of the queries escalation repairs. A descriptor that separates the\n"
        "second and not the first is a rung selector, not a value ranker."
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
