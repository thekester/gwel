import numpy as np
import pytest
from PIL import Image

from gwel.modeling.perturbations import BorderPerturbation


def _image(size: int = 100) -> Image.Image:
    return Image.new("RGB", (size, size), (128, 128, 128))


def test_centre_is_left_untouched() -> None:
    perturbed = BorderPerturbation(band_fraction=0.1, strength=1.0).apply(_image())
    array = np.asarray(perturbed)
    centre = array[30:70, 30:70]
    assert np.all(centre == 128)


def test_border_is_changed() -> None:
    perturbed = BorderPerturbation(band_fraction=0.1, strength=1.0).apply(_image())
    array = np.asarray(perturbed)
    assert not np.all(array[:5, :] == 128)


def test_zero_strength_is_a_no_op() -> None:
    original = _image()
    perturbed = BorderPerturbation(band_fraction=0.2, strength=0.0).apply(original)
    assert np.array_equal(np.asarray(original), np.asarray(perturbed))


def test_perturbation_is_deterministic_given_a_seed() -> None:
    first = BorderPerturbation(seed=7).apply(_image())
    second = BorderPerturbation(seed=7).apply(_image())
    assert np.array_equal(np.asarray(first), np.asarray(second))
    third = BorderPerturbation(seed=8).apply(_image())
    assert not np.array_equal(np.asarray(first), np.asarray(third))


def test_size_and_mode_are_preserved() -> None:
    original = Image.new("RGB", (120, 80), (10, 20, 30))
    perturbed = BorderPerturbation().apply(original)
    assert perturbed.size == original.size
    assert perturbed.mode == "RGB"


def test_area_fraction_grows_with_the_band() -> None:
    assert BorderPerturbation(band_fraction=0.05).area_fraction < BorderPerturbation(
        band_fraction=0.2
    ).area_fraction


def test_invalid_parameters_are_rejected() -> None:
    with pytest.raises(ValueError):
        BorderPerturbation(band_fraction=0.0)
    with pytest.raises(ValueError):
        BorderPerturbation(band_fraction=0.6)
    with pytest.raises(ValueError):
        BorderPerturbation(strength=1.5)
    with pytest.raises(ValueError):
        BorderPerturbation(mode="rainbow")


def test_flat_mode_removes_texture_where_noise_adds_it() -> None:
    original = Image.fromarray(
        np.random.default_rng(0).integers(0, 256, (100, 100, 3), dtype=np.uint8)
    )
    noisy = np.asarray(BorderPerturbation(mode="noise").apply(original), dtype=float)
    flat = np.asarray(BorderPerturbation(mode="flat").apply(original), dtype=float)
    # Variance inside the top band: flat should be near zero, noise should not.
    assert flat[:10, :].std() < 1.0
    assert noisy[:10, :].std() > 10.0


def test_flat_mode_still_preserves_the_centre() -> None:
    original = Image.new("RGB", (100, 100), (200, 50, 50))
    flat = np.asarray(BorderPerturbation(mode="flat", band_fraction=0.1).apply(original))
    assert np.all(flat[30:70, 30:70] == (200, 50, 50))
