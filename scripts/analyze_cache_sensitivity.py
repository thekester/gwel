"""Does KV-cache reuse refund the probe? A sensitivity analysis with a floor.

The most serious threat to this paper's cost model. VLCache (arXiv 2512.12977)
reports $1.2\\times$-$16\\times$ time-to-first-token speedups by reusing encoder
and KV caches across multimodal requests, and NVIDIA reports up to $28\\times$ in
multi-turn conversations about the same image. Our accounting charges an
escalated query the probe *and* the escalation with no reuse at all. If a
deployed stack refunds most of that, the probe's advantage could evaporate.

Two facts settle it, and both are checkable rather than argued.

**Escalation is a cache miss by construction.** VLCache keys its cache on a hash
over the input pixels: "if this hash matches a previously processed request, the
precomputed image embeddings are retrieved ... entirely bypassing ViT
computation". Escalating means submitting *different pixels*, more of them, so the hash differs and the encoder runs. The reuse literature refunds
repetition; escalation is the opposite of repetition.

**And even a perfect cache cannot close the gap, because decode is
uncacheable.** This is the quantitative claim, and it needs no assumption about
what a serving stack achieves. Reading output entropy requires the cheap pass to
*generate*. Generation is what produces the answer, it differs per query, and no
cache refunds it. Reading the probe stops mid-prefill. So as the cacheable
fraction of a pass goes to one, the probe's saving per escalated query falls to
the decode time and stops there.

Writing ``c`` for the fraction of encoder and prefill cost that a cache refunds,
the saving per escalated query is

    S(c) = (1 - c) * (1 - l/L) * t_prefill  +  t_decode,

whose floor at ``c = 1`` is ``t_decode``. This script evaluates it on the
measured component split.

Usage: python scripts/analyze_cache_sensitivity.py
"""

import argparse
import json
from pathlib import Path

import numpy as np

LAYERS = 32
PROBE_LAYER = 6


def components(path: str, config: str = "longest_384") -> dict[str, float]:
    """Measured encoder / prefill / decode split of one pass."""
    rows = {r["config"]: r for r in json.loads(Path(path).read_text())}
    row = rows[config]
    return {
        "encoder": row["vision_encoder_ms"] + row["projector_ms"],
        "prefill": row["prefill_ms"],
        "decode": row["decode_ms"],
        "total": row["total_ms"],
    }


def saving_per_escalation(parts: dict[str, float], refund: float) -> float:
    """Latency the probe saves on one escalated query, at cache refund ``refund``.

    The entropy policy pays the whole cheap pass before it can decide. The probe
    policy pays the encoder plus ``l/L`` of prefill, both cacheable, and then
    abandons. What separates them is the remaining prefill, which a cache can
    refund, plus decode, which it cannot.
    """
    remaining_prefill = (1.0 - PROBE_LAYER / LAYERS) * parts["prefill"]
    return (1.0 - refund) * remaining_prefill + parts["decode"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latency", default="results/component_latency.json")
    parser.add_argument("--out", default="results/cache_sensitivity.json")
    args = parser.parse_args()

    parts = components(args.latency)
    print(
        f"cheap pass at 384 px: encoder {parts['encoder']:.1f} + "
        f"prefill {parts['prefill']:.1f} + decode {parts['decode']:.1f} ms"
    )
    cacheable = (parts["encoder"] + parts["prefill"]) / parts["total"]
    print(
        f"cacheable in principle (encoder + prefill): {cacheable:.0%} of the pass; "
        f"decode is {1 - cacheable:.0%} and is not\n"
    )

    grid = np.linspace(0.0, 1.0, 11)
    rows = []
    print(f"{'cache refund c':>15}{'probe saves / escalation':>26}{'vs no reuse':>13}")
    baseline = saving_per_escalation(parts, 0.0)
    for refund in grid:
        saving = saving_per_escalation(parts, float(refund))
        rows.append({"refund": float(refund), "saving_ms": saving})
        print(f"{refund:>15.1f}{saving:>26.1f}{saving / baseline:>13.0%}")

    floor = saving_per_escalation(parts, 1.0)
    print(
        f"\nfloor at a perfect cache: {floor:.1f} ms per escalated query, "
        f"{floor / baseline:.0%} of the uncached saving."
    )
    print(
        "The probe's advantage is bounded below by decode, which no cache "
        "refunds,\nbecause reading entropy requires generating an answer."
    )

    # The break-even a reviewer would ask for: what refund would be needed to
    # erase the advantage entirely? None, by the above, report it explicitly.
    erases = bool(floor <= 0.0)
    print(f"\nrefund that erases the probe's advantage: "
          f"{'exists' if erases else 'none exists'}")

    results = {
        "components": parts,
        "cacheable_fraction": cacheable,
        "curve": rows,
        "floor_ms": floor,
        "floor_share_of_uncached": floor / baseline,
        "advantage_erasable": erases,
    }
    Path(args.out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
