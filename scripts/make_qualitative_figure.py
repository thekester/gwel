"""Three DocVQA pages, rendered at the resolution each rung actually sees.

The paper's taxonomy (the thumbnail suffices / escalation fixes it / no rung
helps) exists only as percentages. This figure shows one real page per case,
downsampled exactly as the processor would (longest edge to the rung's pixel
budget, Lanczos), then upscaled nearest-neighbour so the reader sees the
degradation the model sees rather than a smooth print rescale.

The three examples are chosen by their recorded outcomes under the paper's
scoring policy, not by hand-picking answers: a layout-level question the
thumbnail already answers, a fine-print axis label that only becomes legible
at 1152 px, and a dense table lookup that fails at every rung including full
resolution. Their shares on the full 1200-page corpus are computed here and
written to an artefact so the caption's numbers stay traced.

Usage: PYTHONPATH=src python scripts/make_qualitative_figure.py
"""

import argparse
import json
import textwrap
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from gwel.data.scoring import ScoringPolicy, rescore_records
from gwel.oracle.records import deduplicate_records, read_records

RUNGS = ("lowres_384", "lowres_768", "lowres_1152", "full")

# example id -> (crop box in original pixels, panel title)
CASES = {
    "docvqa-18": ((90, 95, 670, 295), "the thumbnail suffices"),
    "docvqa-88": ((590, 755, 1055, 870), "escalation fixes it"),
    "docvqa-5": ((388, 265, 1160, 500), "no rung helps"),
}
SHOWN = ("lowres_384", "lowres_1152")
PIXELS = {"lowres_384": 384, "lowres_1152": 1152}
TOKENS = {"lowres_384": 64, "lowres_1152": 640}


def rung_view(image: Image.Image, box: tuple, edge: int) -> Image.Image:
    """Crop of the page as the rung sees it: Lanczos down, nearest back up."""
    scale = edge / max(image.size)
    small = image.resize(
        (round(image.width * scale), round(image.height * scale)), Image.LANCZOS
    )
    crop = small.crop(tuple(round(c * scale) for c in box))
    up = 4 if edge <= 384 else 2
    return crop.resize((crop.width * up, crop.height * up), Image.NEAREST)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default="results/runs/docvqa1200_records.jsonl")
    parser.add_argument("--manifest", default="data/processed/docvqa1200/manifest.jsonl")
    parser.add_argument("--figure", default="paper/figures/qualitative.png")
    parser.add_argument("--out", default="results/qualitative_cases.json")
    args = parser.parse_args()

    grouped: dict[str, dict] = defaultdict(dict)
    for r in rescore_records(
        deduplicate_records(read_records(args.records)), ScoringPolicy()
    ):
        grouped[r.example_id][r.config_id] = r
    paths = {
        row["example_id"]: row["image_path"]
        for row in map(json.loads, open(args.manifest, encoding="utf-8"))
    }

    complete = [e for e in grouped if all(c in grouped[e] for c in RUNGS)]
    ok = {e: {c: grouped[e][c].correct for c in RUNGS} for e in complete}
    n = len(complete)
    shares = {
        "n": n,
        "thumbnail_suffices": sum(ok[e]["lowres_384"] for e in complete) / n,
        "fixed_by_1152": sum(
            not ok[e]["lowres_384"] and ok[e]["lowres_1152"] for e in complete
        )
        / n,
        "no_rung_helps": sum(not any(ok[e].values()) for e in complete) / n,
    }

    fig, axes = plt.subplots(
        2, 3, figsize=(12.5, 6.2), gridspec_kw={"hspace": 0.08, "wspace": 0.06}
    )
    cases = {}
    for col, (eid, (box, title)) in enumerate(CASES.items()):
        page = Image.open(paths[eid]).convert("L")
        byc = grouped[eid]
        for row, rung in enumerate(SHOWN):
            ax = axes[row][col]
            ax.imshow(rung_view(page, box, PIXELS[rung]), cmap="gray", aspect="auto")
            ax.set_xticks([])
            ax.set_yticks([])
            good = byc[rung].correct
            mark, colour = ("✓", "#1a7a2e") if good else ("✗", "#b02418")
            ax.text(
                0.03,
                0.05,
                f"{TOKENS[rung]} tok: “{byc[rung].answer.rstrip('.')}” {mark}",
                transform=ax.transAxes,
                fontsize=10.5,
                color=colour,
                family="DejaVu Sans",
                va="bottom",
                bbox={"facecolor": "white", "alpha": 0.88, "edgecolor": colour},
            )
            for side in ax.spines.values():
                side.set_color(colour)
                side.set_linewidth(2.2)
        q = byc["full"].question.strip()
        q = q if len(q) <= 44 else textwrap.fill(q, 40)
        axes[0][col].set_title(
            f"{title}\n“{q}”", fontsize=11, family="DejaVu Sans", pad=8
        )
        cases[eid] = {
            "question": q,
            "gold": list(byc["full"].gold_answers),
            "answers": {c: byc[c].answer for c in RUNGS},
            "correct": {c: bool(byc[c].correct) for c in RUNGS},
        }
    axes[0][0].set_ylabel("seen at 384 px", fontsize=11)
    axes[1][0].set_ylabel("seen at 1152 px", fontsize=11)

    Path(args.figure).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure, dpi=260, bbox_inches="tight", facecolor="white")
    Path(args.out).write_text(json.dumps({"shares": shares, "cases": cases}, indent=2))
    print(f"wrote {args.figure} and {args.out}")
    print(json.dumps(shares, indent=2))


if __name__ == "__main__":
    main()
