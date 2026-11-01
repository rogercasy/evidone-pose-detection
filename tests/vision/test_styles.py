from collections.abc import Callable

import numpy as np
import pytest

from evidone_pose_detection.config import CONFIG
from evidone_pose_detection.models import Frame
from evidone_pose_detection.vision.styles import blurred, filled, pixelated

Conceal = Callable[[Frame], Frame]

ALL_STYLES: list[Conceal] = [blurred, pixelated, filled]


def gradient(size: int) -> Frame:
    column = np.linspace(0, 255, size, dtype=np.uint8)
    return np.repeat(np.tile(column, (size, 1))[:, :, None], 3, axis=2)


def noise(size: int) -> Frame:
    generator = np.random.default_rng(seed=1234)
    return generator.integers(0, 256, (size, size, 3), dtype=np.uint8)


@pytest.mark.parametrize("style", ALL_STYLES, ids=lambda style: style.__name__)
@pytest.mark.parametrize("size", [9, 64, 240])
def test_a_style_returns_a_patch_of_the_same_shape(style: Conceal, size: int) -> None:
    patch = noise(size)

    concealed = style(patch)

    assert concealed.shape == patch.shape
    assert concealed.dtype == patch.dtype


@pytest.mark.parametrize("style", ALL_STYLES, ids=lambda style: style.__name__)
def test_a_style_destroys_the_detail_it_was_given(style: Conceal) -> None:
    patch = noise(120)

    concealed = style(patch)

    assert float(concealed.std()) < float(patch.std()) * 0.3


@pytest.mark.parametrize("style", ALL_STYLES, ids=lambda style: style.__name__)
def test_a_style_leaves_the_patch_it_was_handed_alone(style: Conceal) -> None:
    patch = noise(64)
    untouched = patch.copy()

    style(patch)

    assert (patch == untouched).all(), "the style wrote through its input"


def test_a_fill_is_one_colour_everywhere() -> None:
    concealed = filled(noise(80))

    assert float(concealed.std()) == 0.0
    assert tuple(int(channel) for channel in concealed[0, 0]) == CONFIG.hiding.fill_colour


@pytest.mark.parametrize("size", [64, 160, 320])
def test_pixel_blocks_are_the_patch_divided_by_the_block_count(size: int) -> None:
    concealed = pixelated(gradient(size))

    row = concealed[size // 2, :, 0].astype(int)
    edges = np.flatnonzero(np.diff(row)) + 1
    widths = np.diff(np.concatenate(([0], edges, [size])))

    assert len(edges) <= CONFIG.hiding.pixel_blocks
    assert float(np.median(widths)) == pytest.approx(size / CONFIG.hiding.pixel_blocks, rel=0.2)


def test_a_bigger_patch_gets_bigger_blocks() -> None:
    def block_width(size: int) -> float:
        row = pixelated(gradient(size))[size // 2, :, 0].astype(int)
        edges = np.flatnonzero(np.diff(row)) + 1
        return float(np.median(np.diff(np.concatenate(([0], edges, [size])))))

    small, large = block_width(80), block_width(320)

    assert large > small * 3, "the blocks did not grow with the patch"


def test_a_blur_keeps_no_hard_edges_where_pixelation_leaves_them() -> None:
    patch = gradient(160)

    steps_from_blur = np.count_nonzero(np.diff(blurred(patch)[80, :, 0].astype(int)))
    steps_from_pixels = np.count_nonzero(np.diff(pixelated(patch)[80, :, 0].astype(int)))

    assert steps_from_blur > steps_from_pixels, "the blur should not be blocky"
