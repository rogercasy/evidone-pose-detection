import numpy as np
import pytest

from evidone_pose_detection.models import (
    BodyRegion,
    BoundingBox,
    COCOKeypoint,
    Frame,
    Keypoint,
    PersonPose,
)
from evidone_pose_detection.rendering import REGION_COLOURS, VISIBILITY_THRESHOLD, draw_poses

WIDTH = 200
HEIGHT = 160
HIDDEN = VISIBILITY_THRESHOLD - 0.01
BLACK = (0, 0, 0)


def blank() -> Frame:
    return np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)


def person_showing(
    visible: dict[COCOKeypoint, tuple[float, float]], confidence: float
) -> PersonPose:
    keypoints = tuple(
        Keypoint(x=visible[name][0], y=visible[name][1], confidence=confidence)
        if name in visible
        else Keypoint(x=0.0, y=0.0, confidence=HIDDEN)
        for name in COCOKeypoint
    )
    return PersonPose(
        box=BoundingBox(left=0.0, top=0.0, right=1.0, bottom=1.0),
        confidence=0.9,
        keypoints=keypoints,
    )


def pixel(frame: Frame, x: int, y: int) -> tuple[int, int, int]:
    blue, green, red = frame[y, x]
    return int(blue), int(green), int(red)


@pytest.mark.parametrize(
    ("name", "region"),
    [
        (COCOKeypoint.NOSE, BodyRegion.HEAD),
        (COCOKeypoint.LEFT_EAR, BodyRegion.HEAD),
        (COCOKeypoint.LEFT_SHOULDER, BodyRegion.TORSO),
        (COCOKeypoint.RIGHT_HIP, BodyRegion.TORSO),
        (COCOKeypoint.LEFT_WRIST, BodyRegion.LIMB),
        (COCOKeypoint.RIGHT_ANKLE, BodyRegion.LIMB),
    ],
)
def test_a_visible_keypoint_draws_a_dot_in_its_region_colour(
    name: COCOKeypoint, region: BodyRegion
) -> None:
    frame = blank()

    draw_poses(frame, [person_showing({name: (80.0, 90.0)}, confidence=0.9)])

    assert pixel(frame, 80, 90) == REGION_COLOURS[region]


def test_a_keypoint_below_the_threshold_draws_nothing() -> None:
    frame = blank()

    draw_poses(frame, [person_showing({COCOKeypoint.NOSE: (80.0, 90.0)}, confidence=HIDDEN)])

    assert not frame.any()


def test_a_keypoint_exactly_at_the_threshold_is_drawn() -> None:
    frame = blank()

    draw_poses(
        frame, [person_showing({COCOKeypoint.NOSE: (80.0, 90.0)}, confidence=VISIBILITY_THRESHOLD)]
    )

    assert pixel(frame, 80, 90) == REGION_COLOURS[BodyRegion.HEAD]


def test_a_bone_is_drawn_when_both_endpoints_are_visible() -> None:
    frame = blank()
    shoulders = {
        COCOKeypoint.LEFT_SHOULDER: (40.0, 100.0),
        COCOKeypoint.RIGHT_SHOULDER: (140.0, 100.0),
    }

    draw_poses(frame, [person_showing(shoulders, confidence=0.9)])

    assert pixel(frame, 90, 100) == REGION_COLOURS[BodyRegion.TORSO]


def test_a_bone_is_skipped_when_one_endpoint_is_hidden() -> None:
    frame = blank()

    draw_poses(frame, [person_showing({COCOKeypoint.LEFT_SHOULDER: (40.0, 100.0)}, confidence=0.9)])

    assert pixel(frame, 90, 100) == BLACK
    assert pixel(frame, 40, 100) == REGION_COLOURS[BodyRegion.TORSO]


def test_limb_bones_use_the_limb_colour_not_the_joint_colour() -> None:
    frame = blank()
    arm = {COCOKeypoint.LEFT_SHOULDER: (40.0, 20.0), COCOKeypoint.LEFT_ELBOW: (140.0, 20.0)}

    draw_poses(frame, [person_showing(arm, confidence=0.9)])

    assert pixel(frame, 90, 20) == REGION_COLOURS[BodyRegion.LIMB]
    assert pixel(frame, 40, 20) == REGION_COLOURS[BodyRegion.TORSO]


def test_head_torso_and_limbs_are_told_apart_by_colour() -> None:
    assert len(set(REGION_COLOURS.values())) == len(BodyRegion)


def test_drawing_mutates_the_frame_it_was_given_and_hands_it_back() -> None:
    frame = blank()

    drawn = draw_poses(frame, [person_showing({COCOKeypoint.NOSE: (80.0, 90.0)}, confidence=0.9)])

    assert drawn is frame
    assert frame.any()


def test_two_people_are_both_drawn() -> None:
    frame = blank()
    people = [
        person_showing({COCOKeypoint.NOSE: (40.0, 40.0)}, confidence=0.9),
        person_showing({COCOKeypoint.NOSE: (160.0, 120.0)}, confidence=0.9),
    ]

    draw_poses(frame, people)

    assert pixel(frame, 40, 40) == REGION_COLOURS[BodyRegion.HEAD]
    assert pixel(frame, 160, 120) == REGION_COLOURS[BodyRegion.HEAD]


def test_a_frame_without_people_is_left_untouched() -> None:
    frame = blank()

    draw_poses(frame, [])

    assert not frame.any()
