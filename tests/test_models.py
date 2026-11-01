import pytest

from evidone_pose_detection.models import (
    DEFAULT_DRAW_SKELETON,
    DEFAULT_HIDE_STYLE,
    KEYPOINT_REGIONS,
    SKELETON,
    CaptureProperties,
    COCOKeypoint,
    HideStyle,
    ViewSettings,
)


@pytest.mark.parametrize(("fps", "expected"), [(30.0, 33), (25.0, 40), (1.0, 1000), (60.0, 17)])
def test_frame_interval_follows_the_capture_rate(fps: float, expected: int) -> None:
    properties = CaptureProperties(width=640, height=480, fps=fps)

    assert properties.frame_interval_ms == expected


# The canonical COCO-17 skeleton, as published with the dataset (converted to 0-indexed).
CANONICAL_COCO_BONES = frozenset(
    frozenset(pair)
    for pair in [
        (15, 13),
        (13, 11),
        (16, 14),
        (14, 12),
        (11, 12),
        (5, 11),
        (6, 12),
        (5, 6),
        (5, 7),
        (6, 8),
        (7, 9),
        (8, 10),
        (1, 2),
        (0, 1),
        (0, 2),
        (1, 3),
        (2, 4),
        (3, 5),
        (4, 6),
    ]
)


def test_the_skeleton_is_the_canonical_coco_one() -> None:
    drawn = frozenset(frozenset((bone.start, bone.end)) for bone in SKELETON)

    assert drawn == CANONICAL_COCO_BONES
    assert len(SKELETON) == len(CANONICAL_COCO_BONES)


def test_no_bone_is_listed_twice() -> None:
    assert len({frozenset((bone.start, bone.end)) for bone in SKELETON}) == len(SKELETON)


@pytest.mark.parametrize("name", list(COCOKeypoint))
def test_every_keypoint_belongs_to_a_region(name: COCOKeypoint) -> None:
    assert name in KEYPOINT_REGIONS


def test_the_app_starts_on_blur_with_the_skeleton_drawn() -> None:
    settings = ViewSettings.defaults()

    assert settings.hide_style is HideStyle.BLUR
    assert settings.hide_style is DEFAULT_HIDE_STYLE
    assert settings.draw_skeleton is DEFAULT_DRAW_SKELETON is True


def test_settings_are_edited_in_place_so_a_reader_sees_the_change() -> None:
    settings = ViewSettings.defaults()
    seen_by_the_loop = settings

    settings.hide_style = HideStyle.FILL
    settings.draw_skeleton = False

    assert seen_by_the_loop.hide_style is HideStyle.FILL
    assert seen_by_the_loop.draw_skeleton is False


def test_each_set_of_defaults_is_independent() -> None:
    first = ViewSettings.defaults()
    second = ViewSettings.defaults()

    first.hide_style = HideStyle.PIXELATE

    assert second.hide_style is HideStyle.BLUR
