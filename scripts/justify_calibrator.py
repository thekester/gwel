"""Isotonic was adopted from UCCI. Is it the right choice here?

The decision rule calibrates a routing score to an expected gain, and the paper
uses isotonic regression because Kotte (arXiv 2605.18796) does. Borrowing a
component without testing it is how a paper inherits someone else's assumptions,
and the two settings differ in the way that matters for calibration: their
target is a probability in [0, 1] with tens of thousands of calibration points,
ours is a signed gain in [-1, +1] with a few hundred.

Three families on identical folds:

  isotonic    monotone, non-parametric, the current choice; free to fit any
              non-decreasing shape but able to interpolate noise at small n
  Platt       a fitted sigmoid, two parameters; cannot represent a plateau but
              cannot chase noise either
  binned      equal-mass bins with the empirical mean in each, the simplest
              thing that could work and a floor the others must clear

Judged on three axes, because a calibrator that ranks well can still be
miscalibrated and a calibrator that is well calibrated can still route badly:
ranking (AUROC), magnitude (calibration error), and the policy it produces.

Usage: PYTHONPATH=scripts python scripts/justify_calibrator.py
"""

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from gwel.config import load_config
from gwel.data.scoring import ScoringPolicy, rescore_records
from gwel.oracle.records import deduplicate_records, read_records
from gwel.oracle.token_cost import fit_token_cost
from gwel.router.decision import (
    fit_gain_calibrator,
    gain_calibration_error,
    signed_gain,
)
from gwel.router.evaluate import auroc, bootstrap_interval

CHEAP = "lowres_384"
ESCALATED = "full"
RESAMPLES = 30
HELDOUT = 300
VALUE = 1600.0
MIN_BUCKET = 50
BINS = 10


@dataclass(frozen=True)
class Platt:
    """Two-parameter sigmoid, fitted by gradient descent on squared error.

    A sigmoid cannot represent the flat regions an isotonic fit can, which is
    exactly the trade being tested: fewer parameters, no ability to chase noise.
    """

    scale: float
    shift: float

    def predict(self, scores: np.ndarray) -> np.ndarray:
        z = self.scale * (np.asarray(scores, dtype=np.float64) - self.shift)
        return 2.0 / (1.0 + np.exp(-z)) - 1.0


def fit_platt(scores: np.ndarray, gains: np.ndarray, *, epochs: int = 600) -> Platt:
    scores = np.asarray(scores, dtype=np.float64)
    gains = np.asarray(gains, dtype=np.float64)
    spread = scores.std() + 1e-8
    scale, shift = 1.0 / spread, float(scores.mean())
    rate = 0.05
    for _ in range(epochs):
        z = scale * (scores - shift)
        sigmoid = 1.0 / (1.0 + np.exp(-z))
        prediction = 2.0 * sigmoid - 1.0
        residual = prediction - gains
        common = residual * 2.0 * sigmoid * (1.0 - sigmoid)
        scale -= rate * float((common * (scores - shift)).mean())
        shift -= rate * float((common * -scale).mean())
    return Platt(scale=scale, shift=shift)


@dataclass(frozen=True)
class Binned:
    """Equal-mass bins holding the empirical mean gain: the simplest baseline."""

    edges: np.ndarray
    values: np.ndarray

    def predict(self, scores: np.ndarray) -> np.ndarray:
        index = np.clip(
            np.searchsorted(self.edges, np.asarray(scores), side="right") - 1,
            0,
            len(self.values) - 1,
        )
        return self.values[index]


def fit_binned(scores: np.ndarray, gains: np.ndarray, *, bins: int = BINS) -> Binned:
    edges = np.quantile(scores, np.linspace(0, 1, bins + 1))[:-1]
    index = np.clip(np.searchsorted(edges, scores, side="right") - 1, 0, bins - 1)
    values = np.array(
        [gains[index == b].mean() if (index == b).any() else 0.0 for b in range(bins)]
    )
    return Binned(edges=edges, values=values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/docvqa1200.yaml")
    parser.add_argument("--out", default="results/calibrator_family.json")
    args = parser.parse_args()

    config = load_config(args.config)
    grouped: dict[str, dict] = defaultdict(dict)
    for record in rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    ):
        grouped[record.example_id][record.config_id] = record
    ids = [
        e
        for e in grouped
        if CHEAP in grouped[e] and ESCALATED in grouped[e] and grouped[e][CHEAP].signals
    ]
    correct = {c: np.array([grouped[e][c].correct for e in ids]) for c in (CHEAP, ESCALATED)}
    entropy = np.array([float(grouped[e][CHEAP].signals["mean_entropy"]) for e in ids])
    gains = signed_gain(correct[CHEAP], correct[ESCALATED])

    by: dict[int, list[float]] = defaultdict(list)
    for e in ids:
        for record in grouped[e].values():
            by[int(record.visual_tokens)].append(record.latency_ms)
    good = [t for t in sorted(by) if len(by[t]) >= MIN_BUCKET]
    model = fit_token_cost(good, [float(np.median(by[t])) for t in good])
    latency = {
        c: model.predict(np.array([grouped[e][c].visual_tokens for e in ids], float))
        for c in (CHEAP, ESCALATED)
    }
    delta = latency[ESCALATED] - latency[CHEAP]
    tau = float(delta.mean()) / VALUE

    families = {
        "isotonic": lambda s, g: fit_gain_calibrator(s, g),
        "Platt sigmoid": fit_platt,
        "equal-mass bins": fit_binned,
    }
    collected: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"auroc": [], "error": [], "accuracy": [], "latency": []}
    )

    for seed in range(RESAMPLES):
        rng = np.random.default_rng(11000 + seed)
        shuffled = rng.permutation(len(ids))
        test, train = shuffled[:HELDOUT], shuffled[HELDOUT:]
        truth = [bool(x) for x in (gains[test] > 0)]
        if len(set(truth)) < 2:
            continue
        for name, fit in families.items():
            calibrator = fit(entropy[train], gains[train])
            predicted = calibrator.predict(entropy[test])
            fires = predicted > tau
            collected[name]["auroc"].append(auroc(predicted.tolist(), truth))
            collected[name]["error"].append(gain_calibration_error(predicted, gains[test]))
            collected[name]["accuracy"].append(
                float(np.where(fires, correct[ESCALATED][test], correct[CHEAP][test]).mean())
            )
            collected[name]["latency"].append(
                float(np.where(fires, latency[ESCALATED][test], latency[CHEAP][test]).mean())
            )

    print(f"n={len(ids)}, {RESAMPLES} resamples, break-even tau={tau:.3f} at V={VALUE:.0f}\n")
    print(f"{'calibrator':<20}{'AUROC':>22}{'calib. error':>16}{'accuracy':>10}{'latency':>10}")
    rows = {}
    for name, block in collected.items():
        ranking = bootstrap_interval(block["auroc"])
        error = float(np.mean(block["error"]))
        rows[name] = {
            "auroc": [ranking.estimate, ranking.low, ranking.high],
            "calibration_error": error,
            "accuracy": float(np.mean(block["accuracy"])),
            "latency": float(np.mean(block["latency"])),
        }
        print(
            f"{name:<20}{str(ranking):>22}{error:>16.4f}"
            f"{np.mean(block['accuracy']):>10.3f}{np.mean(block['latency']):>10.1f}"
        )

    # Paired against the current choice, since the question is whether to keep it.
    print()
    reference = "isotonic"
    for name in families:
        if name == reference:
            continue
        acc = bootstrap_interval(
            [a - b for a, b in zip(collected[name]["accuracy"],
                                   collected[reference]["accuracy"])]
        )
        lat = bootstrap_interval(
            [a - b for a, b in zip(collected[name]["latency"],
                                   collected[reference]["latency"])]
        )
        rows[name]["accuracy_delta"] = [acc.estimate, acc.low, acc.high]
        rows[name]["latency_delta"] = [lat.estimate, lat.low, lat.high]
        print(f"  {name} minus isotonic: accuracy {acc}, latency {lat} ms")

    best_error = min(rows, key=lambda k: rows[k]["calibration_error"])
    print(
        f"\nlowest calibration error: {best_error}. Ranking is nearly identical across"
        "\nfamilies, which is the point: the calibrator is chosen for its magnitudes,"
        "\nsince the rule thresholds a magnitude rather than a rank."
    )
    Path(args.out).write_text(json.dumps({"tau": tau, "families": rows}, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
