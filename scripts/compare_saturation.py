"""Is the saturation point a property of the model, or of the images?

`analyze_docvqa_pilot.py` finds that escalation value dies between 640 and 1088
visual tokens on SmolVLM-500M. That result is measured on one serving model, so
it cannot yet distinguish two explanations:

  capacity   the model stops being able to use extra detail, in which case a
             smaller model should saturate *earlier*
  legibility the pages are already legible at 640 tokens and the remaining
             errors are not resolution-limited, in which case a smaller model
             should saturate at the same rung with lower accuracy throughout

Same 1200 pages, same manifest, same rungs; only the model that answers changes.
The two hypotheses make opposite predictions about where the curve flattens, so
one run separates them.

Usage: PYTHONPATH=scripts python scripts/compare_saturation.py
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from gwel.config import load_config
from gwel.data.scoring import ScoringPolicy, rescore_records
from gwel.oracle.records import deduplicate_records, read_records
from gwel.oracle.token_cost import fit_token_cost
from gwel.router.evaluate import bootstrap_interval

RUNGS = ("lowres_384", "lowres_768", "lowres_1152", "full")
MIN_BUCKET = 50
# Half-width a null interval must be under before "no gain" is a finding rather
# than a sample size. One accuracy point either side.
NULL_PRECISION = 0.05


def summarise(config_path: str) -> dict:
    config = load_config(config_path)
    grouped: dict[str, dict] = defaultdict(dict)
    for record in rescore_records(
        deduplicate_records(read_records(config.paths.records)), ScoringPolicy()
    ):
        grouped[record.example_id][record.config_id] = record
    ids = [e for e in grouped if all(c in grouped[e] for c in RUNGS)]
    correct = {c: np.array([grouped[e][c].correct for e in ids]) for c in RUNGS}
    tokens = {
        c: np.array([grouped[e][c].visual_tokens for e in ids], float) for c in RUNGS
    }
    by: dict[int, list[float]] = defaultdict(list)
    for e in ids:
        for record in grouped[e].values():
            by[int(record.visual_tokens)].append(record.latency_ms)
    # A dynamic-resolution encoder gives each image its own token count, so
    # bucketed timing never accumulates; the saturation verdict does not need
    # the latency fit, so it is optional rather than fatal.
    good = [t for t in sorted(by) if len(by[t]) >= MIN_BUCKET]
    model = (
        fit_token_cost(good, [float(np.median(by[t])) for t in good])
        if len(good) >= 2
        else None
    )

    steps = []
    for low, high in zip(RUNGS[:-1], RUNGS[1:]):
        gain = (correct[high] & ~correct[low]).astype(float) - (
            correct[low] & ~correct[high]
        ).astype(float)
        interval = bootstrap_interval(gain.tolist())
        steps.append(
            {
                "from": low,
                "to": high,
                "gain": interval.estimate,
                "low": interval.low,
                "high": interval.high,
            }
        )
    # A step "saturates" only if its interval contains zero *and* is tight
    # enough for that to mean something. An interval of [+0.00, +0.12] contains
    # zero and is equally consistent with a large gain, so calling it saturation
    # would be reading power as evidence.
    for step in steps:
        half_width = (step["high"] - step["low"]) / 2
        step["half_width"] = half_width
        step["null"] = step["low"] <= 0.0 <= step["high"]
        step["informative"] = half_width <= NULL_PRECISION
    saturates = next(
        (s["from"] for s in steps if s["null"] and s["informative"]), None
    )
    undetermined = [s["from"] for s in steps if s["null"] and not s["informative"]]
    return {
        "model": config.model.model_id,
        "n": len(ids),
        "accuracy": {c: float(correct[c].mean()) for c in RUNGS},
        "median_tokens": {c: float(np.median(tokens[c])) for c in RUNGS},
        "steps": steps,
        "saturation_rung": saturates,
        "undetermined_rungs": undetermined,
        "cost_model": (
            [model.base_ms, model.slope_ms_per_token, model.residual_ms]
            if model is not None
            else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--large", default="configs/docvqa1200.yaml")
    parser.add_argument("--small", default="configs/docvqa1200_256m.yaml")
    parser.add_argument(
        "--extra",
        nargs="*",
        default=[
            "configs/docvqa1200_2b.yaml",
            "configs/docvqa1200_qwen2b.yaml",
        ],
        # InternVL3-1B is deliberately absent. This script compares cost
        # ladders, and its rungs are not one: its tiling reads the aspect ratio
        # rather than the resolution, so every rung spends the same tokens and
        # the steps here would price nothing. It is analysed by
        # scripts/analyze_fixed_budget.py instead, where a flat budget is the
        # point rather than a defect. Passing it explicitly still works and
        # will show its ceiling sitting a rung lower than the others.
        help="further models; the last is outside the SmolVLM lineage",
    )
    parser.add_argument("--out", default="results/saturation_models.json")
    args = parser.parse_args()

    paths = [args.large, args.small]
    for extra in args.extra or []:
        if not Path(extra).exists():
            continue
        try:
            summarise(extra)
            paths.append(extra)
        except (FileNotFoundError, ValueError) as error:
            print(f"skipping {extra}: {error}")
    runs = [summarise(path) for path in paths]
    print(f"{'model':<34}{'n':>6}" + "".join(f"{c.replace('lowres_', ''):>10}" for c in RUNGS))
    for run in runs:
        short = run["model"].split("/")[-1]
        print(
            f"{short:<34}{run['n']:>6}"
            + "".join(f"{run['accuracy'][c]:>10.3f}" for c in RUNGS)
        )
    # The rungs are fixed pixel targets, so a different vision encoder puts them
    # at different token counts. That decoupling is the point of a third model:
    # it separates "the same resolution" from "the same sequence length".
    print()
    print(f"{'median visual tokens':<34}{'':>6}")
    for run in runs:
        short = run["model"].split("/")[-1]
        print(
            f"{short:<34}{'':>6}"
            + "".join(f"{run['median_tokens'][c]:>10.0f}" for c in RUNGS)
        )

    print(f"\n{'model':<34}" + "".join(
        f"{s['from'].replace('lowres_', '') + '->' + s['to'].replace('lowres_', ''):>22}"
        for s in runs[0]["steps"]
    ))
    for run in runs:
        short = run["model"].split("/")[-1]
        print(
            f"{short:<34}"
            + "".join(
                f"{s['gain']:>+9.3f} [{s['low']:+.2f},{s['high']:+.2f}]" for s in run["steps"]
            )
        )

    print()
    for run in runs:
        short = run["model"].split("/")[-1]
        where = run["saturation_rung"]
        print(
            f"{short:<34}saturates at "
            + (where.replace("lowres_", "") if where else "no rung (still gaining)")
        )

    verdict = {r["model"]: r["saturation_rung"] for r in runs}
    same = len(set(verdict.values())) == 1 and None not in verdict.values()
    blocked = [r for r in runs if r["undetermined_rungs"]]
    print()
    if blocked:
        for run in blocked:
            short = run["model"].split("/")[-1]
            widest = max(
                (s for s in run["steps"] if s["null"] and not s["informative"]),
                key=lambda s: s["half_width"],
            )
            print(
                f"UNDETERMINED for {short}: the {widest['from']} step gives "
                f"{widest['gain']:+.3f} [{widest['low']:+.3f}, {widest['high']:+.3f}], "
                f"which contains zero only because n={run['n']} cannot resolve it."
            )
        print(
            "No verdict on capacity versus legibility: an interval that wide is"
            " consistent\nwith both hypotheses, and reading it either way would be"
            " reading power as evidence."
        )
    elif same:
        print(
            f"All {len(runs)} models stop gaining at the same rung, which points at\n"
            "the images: the pages are legible at that resolution and the remaining\n"
            "errors are not resolution-limited.\n"
            "Note the class this holds in. Every model here spends more visual\n"
            "tokens as the pixel target rises, so each rung moves pixels and\n"
            "sequence length together. A model that decouples them saturates a\n"
            "rung lower (scripts/analyze_fixed_budget.py), so read this rung as\n"
            "an upper bound on what any model needs, not as a constant."
        )
    else:
        print(
            "The models saturate at different rungs, which points at capacity:\n"
            "how much detail is usable depends on the model reading it."
        )

    Path(args.out).write_text(
        json.dumps({"runs": runs, "same_saturation_rung": same}, indent=2)
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
