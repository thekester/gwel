"""Image operations backing the visual actions: downscaling and crop grids."""

from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class CropBox:
    """Pixel-space crop rectangle within the full-resolution image."""

    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    def to_dict(self) -> dict[str, int]:
        return {"left": self.left, "top": self.top, "right": self.right, "bottom": self.bottom}


def downscale(image: Image.Image, longest_side: int) -> Image.Image:
    """Resize so the longest side is at most ``longest_side``, keeping aspect."""
    if longest_side < 1:
        raise ValueError("longest_side must be >= 1")
    width, height = image.size
    scale = longest_side / max(width, height)
    if scale >= 1.0:
        return image.copy()
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.Resampling.LANCZOS)


def crop_grid(
    width: int,
    height: int,
    *,
    rows: int,
    cols: int,
    overlap: float = 0.0,
) -> list[CropBox]:
    """Return an ``rows x cols`` grid of crop boxes with fractional overlap.

    Each cell is expanded by ``overlap`` times its size on every inner edge,
    clamped to the image bounds. Boxes are ordered row-major.
    """
    if rows < 1 or cols < 1:
        raise ValueError("rows and cols must be >= 1")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("overlap must be in [0, 1)")

    cell_w = width / cols
    cell_h = height / rows
    pad_w = cell_w * overlap / 2.0
    pad_h = cell_h * overlap / 2.0

    boxes: list[CropBox] = []
    for row in range(rows):
        for col in range(cols):
            left = max(0, round(col * cell_w - pad_w))
            top = max(0, round(row * cell_h - pad_h))
            right = min(width, round((col + 1) * cell_w + pad_w))
            bottom = min(height, round((row + 1) * cell_h + pad_h))
            boxes.append(CropBox(left=left, top=top, right=right, bottom=bottom))
    return boxes


def extract_crop(image: Image.Image, box: CropBox, *, longest_side: int) -> Image.Image:
    """Crop ``box`` from the full-resolution image and cap its longest side."""
    crop = image.crop((box.left, box.top, box.right, box.bottom))
    return downscale(crop, longest_side)
