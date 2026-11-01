import numpy as np
import pytest

from evidone_pose_detection.models import Frame, FrameStats
from evidone_pose_detection.vision.overlay import (
    PANEL_OPACITY,
    PANEL_ORIGIN,
    TEXT_COLOUR,
    draw_stats,
)

WIDTH = 480
HEIGHT = 320
STATS = FrameStats(inference_ms=18.4, frames_per_second=27.3, people=2)


def flat(value: int) -> Frame:
    return np.full((HEIGHT, WIDTH, 3), value, dtype=np.uint8)


def panel_corner(frame: Frame) -> tuple[int, int, int]:
    left, top = PANEL_ORIGIN
    blue, green, red = frame[top + 2, left + 2]
    return int(blue), int(green), int(red)


def test_the_panel_lands_in_the_top_left_corner() -> None:
    frame = flat(200)

    draw_stats(frame, STATS)

    assert panel_corner(frame) != (200, 200, 200)
    assert (frame[HEIGHT - 5 :, WIDTH - 5 :] == 200).all(), "the bottom right was touched"
    assert (frame[0, 0] == 200).all(), "the panel should be inset, not flush to the edge"


def test_the_panel_is_translucent_not_opaque() -> None:
    bright = flat(255)
    dark = flat(0)

    draw_stats(bright, STATS)
    draw_stats(dark, STATS)

    # The same panel over different video must give different pixels, or it is opaque.
    assert panel_corner(bright) != panel_corner(dark)


def test_the_backdrop_keeps_a_readable_share_of_the_video() -> None:
    frame = flat(200)

    draw_stats(frame, STATS)

    shaded = panel_corner(frame)[0]
    assert shaded < 200, "the backdrop did not darken the video"
    assert shaded == pytest.approx(200 * (1.0 - PANEL_OPACITY), abs=30)


def test_the_three_numbers_are_written_on_the_frame() -> None:
    frame = flat(0)

    draw_stats(frame, STATS)

    text_pixels = int(np.count_nonzero((frame == TEXT_COLOUR).all(axis=2)))
    assert text_pixels > 100, "no text was drawn"


@pytest.mark.parametrize(
    "stats",
    [
        FrameStats(inference_ms=0.0, frames_per_second=0.0, people=0),
        FrameStats(inference_ms=9999.9, frames_per_second=999.9, people=128),
    ],
)
def test_extreme_numbers_still_draw(stats: FrameStats) -> None:
    frame = flat(120)

    draw_stats(frame, stats)

    assert panel_corner(frame) != (120, 120, 120)


def test_a_frame_too_small_for_the_panel_is_left_alone() -> None:
    frame = np.full((6, 6, 3), 90, dtype=np.uint8)

    drawn = draw_stats(frame, STATS)

    assert drawn is frame
    assert (frame == 90).all()


def test_drawing_mutates_the_frame_and_hands_it_back() -> None:
    frame = flat(200)

    drawn = draw_stats(frame, STATS)

    assert drawn is frame
