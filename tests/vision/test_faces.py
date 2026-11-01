import math
from collections.abc import Mapping

import pytest

from evidone_pose_detection.config import CONFIG
from evidone_pose_detection.models import (
    FACE_LANDMARKS,
    BoundingBox,
    COCOKeypoint,
    FaceRegion,
    Keypoint,
    PersonPose,
)
from evidone_pose_detection.vision.faces import face_region
from evidone_pose_detection.vision.visibility import is_visible

SEEN = 0.9
UNSEEN = CONFIG.faces.visibility_threshold - 0.01

Points = Mapping[COCOKeypoint, tuple[float, float, float]]

FACING_CAMERA: Points = {
    COCOKeypoint.NOSE: (100.0, 100.0, SEEN),
    COCOKeypoint.LEFT_EYE: (90.0, 92.0, SEEN),
    COCOKeypoint.RIGHT_EYE: (110.0, 92.0, SEEN),
    COCOKeypoint.LEFT_EAR: (78.0, 96.0, SEEN),
    COCOKeypoint.RIGHT_EAR: (122.0, 96.0, SEEN),
}

# Turned side on: the far eye and far ear are occluded, so the eye-to-eye and
# ear-to-ear spans both collapse.
IN_PROFILE: Points = {
    COCOKeypoint.NOSE: (100.0, 100.0, SEEN),
    COCOKeypoint.LEFT_EYE: (92.0, 94.0, SEEN),
    COCOKeypoint.LEFT_EAR: (76.0, 98.0, SEEN),
    COCOKeypoint.RIGHT_EYE: (108.0, 94.0, UNSEEN),
    COCOKeypoint.RIGHT_EAR: (124.0, 98.0, UNSEEN),
}

# Back of the head: nose and both eyes are gone, only the ears survive.
FACING_AWAY: Points = {
    COCOKeypoint.NOSE: (100.0, 100.0, UNSEEN),
    COCOKeypoint.LEFT_EYE: (90.0, 92.0, UNSEEN),
    COCOKeypoint.RIGHT_EYE: (110.0, 92.0, UNSEEN),
    COCOKeypoint.LEFT_EAR: (82.0, 96.0, SEEN),
    COCOKeypoint.RIGHT_EAR: (118.0, 96.0, SEEN),
}

NO_FACE_AT_ALL: Points = {name: (100.0, 100.0, UNSEEN) for name in FACE_LANDMARKS} | {
    COCOKeypoint.LEFT_SHOULDER: (60.0, 160.0, SEEN),
    COCOKeypoint.RIGHT_SHOULDER: (140.0, 160.0, SEEN),
}


def person_with(points: Points) -> PersonPose:
    keypoints = tuple(
        Keypoint(*points[name]) if name in points else Keypoint(0.0, 0.0, 0.0)
        for name in COCOKeypoint
    )
    return PersonPose(
        box=BoundingBox(left=0.0, top=0.0, right=200.0, bottom=200.0),
        confidence=0.9,
        keypoints=keypoints,
    )


def covers(region: FaceRegion, keypoint: Keypoint) -> bool:
    distance = math.hypot(keypoint.x - region.centre_x, keypoint.y - region.centre_y)
    return distance <= region.radius


@pytest.mark.parametrize(
    ("name", "points"),
    [("facing camera", FACING_CAMERA), ("in profile", IN_PROFILE), ("facing away", FACING_AWAY)],
)
def test_a_partly_visible_face_still_produces_a_region(name: str, points: Points) -> None:
    region = face_region(person_with(points))

    assert region is not None
    assert region.radius > 0.0


@pytest.mark.parametrize(
    ("name", "points"),
    [("facing camera", FACING_CAMERA), ("in profile", IN_PROFILE), ("facing away", FACING_AWAY)],
)
def test_the_region_covers_every_visible_face_keypoint(name: str, points: Points) -> None:
    person = person_with(points)
    region = face_region(person)

    assert region is not None
    for landmark in FACE_LANDMARKS:
        keypoint = person.keypoint(landmark)
        if is_visible(keypoint):
            assert covers(region, keypoint), f"{landmark.name} escaped the region"


def test_a_face_with_no_visible_keypoints_has_no_region() -> None:
    assert face_region(person_with(NO_FACE_AT_ALL)) is None


def test_a_person_with_no_keypoints_at_all_has_no_region() -> None:
    assert face_region(person_with({})) is None


def test_a_hidden_keypoint_is_ignored_even_where_it_would_enlarge_the_region() -> None:
    far_away = dict(IN_PROFILE) | {COCOKeypoint.RIGHT_EAR: (900.0, 900.0, UNSEEN)}

    region = face_region(person_with(far_away))
    baseline = face_region(person_with(IN_PROFILE))

    assert region == baseline


def test_the_largest_span_wins_when_the_ears_are_the_widest_measure() -> None:
    wide_ears = dict(FACING_CAMERA) | {
        COCOKeypoint.LEFT_EAR: (60.0, 96.0, SEEN),
        COCOKeypoint.RIGHT_EAR: (140.0, 96.0, SEEN),
    }

    region = face_region(person_with(wide_ears))

    assert region is not None
    assert region.radius == pytest.approx(80.0 * 0.9)


def test_the_largest_span_wins_when_the_eyes_are_the_widest_measure() -> None:
    narrow_ears = dict(FACING_CAMERA) | {
        COCOKeypoint.LEFT_EAR: (96.0, 96.0, SEEN),
        COCOKeypoint.RIGHT_EAR: (104.0, 96.0, SEEN),
    }

    region = face_region(person_with(narrow_ears))

    assert region is not None
    assert region.radius == pytest.approx(20.0 * 2.0)


def test_a_profile_is_sized_from_the_spans_that_survive() -> None:
    region = face_region(person_with(IN_PROFILE))

    assert region is not None
    nose_to_ear = math.hypot(100.0 - 76.0, 100.0 - 98.0) * 1.2
    assert region.radius >= nose_to_ear


def test_a_lone_landmark_falls_back_to_the_shoulders() -> None:
    lone_nose = {
        COCOKeypoint.NOSE: (100.0, 100.0, SEEN),
        COCOKeypoint.LEFT_SHOULDER: (60.0, 160.0, SEEN),
        COCOKeypoint.RIGHT_SHOULDER: (140.0, 160.0, SEEN),
    }

    region = face_region(person_with(lone_nose))

    assert region is not None
    assert region.radius == pytest.approx(80.0 * CONFIG.faces.shoulder_scale)
    assert (region.centre_x, region.centre_y) == (100.0, 100.0)


def test_a_lone_landmark_without_shoulders_has_no_region() -> None:
    assert face_region(person_with({COCOKeypoint.NOSE: (100.0, 100.0, SEEN)})) is None


def test_a_keypoint_exactly_at_the_threshold_counts_as_visible() -> None:
    borderline = {
        COCOKeypoint.LEFT_EYE: (90.0, 92.0, CONFIG.faces.visibility_threshold),
        COCOKeypoint.RIGHT_EYE: (110.0, 92.0, CONFIG.faces.visibility_threshold),
    }

    assert face_region(person_with(borderline)) is not None


@pytest.mark.parametrize("scale", [0.5, 2.0, 5.0])
def test_the_region_scales_with_the_head(scale: float) -> None:
    scaled = {
        name: (100.0 + (x - 100.0) * scale, 100.0 + (y - 100.0) * scale, confidence)
        for name, (x, y, confidence) in FACING_CAMERA.items()
    }

    baseline = face_region(person_with(FACING_CAMERA))
    grown = face_region(person_with(scaled))

    assert baseline is not None
    assert grown is not None
    assert grown.radius == pytest.approx(baseline.radius * scale)


def test_every_face_span_uses_two_different_face_landmarks() -> None:
    for span in CONFIG.faces.spans:
        assert span.start in FACE_LANDMARKS
        assert span.end in FACE_LANDMARKS
        assert span.start != span.end


def test_the_spans_are_independent_measurements() -> None:
    pairs = {frozenset((span.start, span.end)) for span in CONFIG.faces.spans}

    assert len(pairs) == len(CONFIG.faces.spans)
    assert len(CONFIG.faces.spans) >= 4


def test_coverage_margin_grows_the_region_beyond_the_landmarks() -> None:
    assert CONFIG.faces.landmark_coverage_margin > 1.0
