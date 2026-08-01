import pytest
from PIL import Image

from gwel.modeling.imaging import crop_grid, downscale, extract_crop


def test_downscale_caps_longest_side_and_keeps_aspect() -> None:
    image = Image.new("RGB", (1000, 500))
    small = downscale(image, 200)
    assert small.size == (200, 100)


def test_downscale_never_upscales() -> None:
    image = Image.new("RGB", (100, 50))
    assert downscale(image, 200).size == (100, 50)


def test_crop_grid_covers_image_without_overlap() -> None:
    boxes = crop_grid(400, 200, rows=2, cols=2, overlap=0.0)
    assert len(boxes) == 4
    assert boxes[0].to_dict() == {"left": 0, "top": 0, "right": 200, "bottom": 100}
    assert boxes[-1].to_dict() == {"left": 200, "top": 100, "right": 400, "bottom": 200}


def test_crop_grid_overlap_expands_cells_within_bounds() -> None:
    boxes = crop_grid(400, 200, rows=2, cols=2, overlap=0.2)
    for box in boxes:
        assert 0 <= box.left < box.right <= 400
        assert 0 <= box.top < box.bottom <= 200
    assert boxes[0].right > 200  # expanded past the no-overlap boundary


def test_crop_grid_validates_arguments() -> None:
    with pytest.raises(ValueError):
        crop_grid(100, 100, rows=0, cols=1)
    with pytest.raises(ValueError):
        crop_grid(100, 100, rows=1, cols=1, overlap=1.0)


def test_extract_crop_resizes_to_longest_side() -> None:
    image = Image.new("RGB", (1000, 1000))
    boxes = crop_grid(1000, 1000, rows=2, cols=2)
    crop = extract_crop(image, boxes[0], longest_side=128)
    assert max(crop.size) == 128
