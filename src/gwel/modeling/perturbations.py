"""Border perturbations, as a cheap proxy for cascade deferral attacks.

Liu et al. (arXiv 2606.15308) force multimodal cascades to defer by learning a
universal trigger confined to a border region around the image, leaving the
central content intact so the semantics survive. Their trigger is optimised
against a temperature-flattened objective; these perturbations are the
unoptimised version of the same idea, which is enough to ask whether a signal
*can* be moved from the border at all.

An allocation signal that a border band can inflate is one an adversary can use
to shift compute cost onto the provider without ever making the answer wrong.
"""

from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class BorderPerturbation:
    """A frame around the image is replaced; the centre is left untouched.

    ``mode`` selects which attack direction is being probed. Liu et al. force
    *more* escalation to inflate the provider's bill; Sun et al. (arXiv
    2605.17288) point out the opposite threat for text cascades, suppressing
    escalation so the system keeps answering from the weak model, degrading
    accuracy rather than cost. Both directions need a perturbation, and they
    plausibly need different ones.

    - ``noise``: high-frequency random pixels, the pattern a learned trigger
      would occupy, expected to raise uncertainty.
    - ``flat``: a uniform fill, removing information rather than adding
      texture, probed for the suppression direction.
    """

    band_fraction: float = 0.1  # band width as a fraction of the shorter side
    strength: float = 1.0       # 0 leaves the image alone, 1 fully replaces the band
    mode: str = "noise"
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.band_fraction <= 0.5:
            raise ValueError("band_fraction must be in (0, 0.5]")
        if not 0.0 <= self.strength <= 1.0:
            raise ValueError("strength must be in [0, 1]")
        if self.mode not in ("noise", "flat"):
            raise ValueError(f"unknown mode {self.mode!r}")

    def _band_mask(self, height: int, width: int) -> np.ndarray:
        band = max(1, int(round(min(height, width) * self.band_fraction)))
        mask = np.zeros((height, width), dtype=np.float32)
        mask[:band, :] = 1.0
        mask[-band:, :] = 1.0
        mask[:, :band] = 1.0
        mask[:, -band:] = 1.0
        return mask

    def apply(self, image: Image.Image) -> Image.Image:
        """Return a copy with the border replaced according to ``mode``."""
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
        height, width = array.shape[:2]
        mask = self._band_mask(height, width)

        if self.mode == "noise":
            rng = np.random.default_rng(self.seed)
            replacement = rng.integers(0, 256, size=array.shape).astype(np.float32)
        else:
            # Fill with the image's own mean, so the band carries no texture
            # and no colour discontinuity to attract attention.
            replacement = np.broadcast_to(
                array.reshape(-1, 3).mean(axis=0), array.shape
            ).astype(np.float32)

        blend = (mask * self.strength)[:, :, None]
        perturbed = array * (1.0 - blend) + replacement * blend
        return Image.fromarray(perturbed.clip(0, 255).astype(np.uint8))

    @property
    def area_fraction(self) -> float:
        """Share of a square image the band covers, for reporting."""
        band = self.band_fraction
        return 1.0 - (1.0 - 2.0 * band) ** 2 if band < 0.5 else 1.0
