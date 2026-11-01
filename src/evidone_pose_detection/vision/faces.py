import math
from collections.abc import Sequence

from evidone_pose_detection.models import (
    FACE_LANDMARKS,
    FACE_SPANS,
    SHOULDER_SPAN,
    FaceRegion,
    FaceSpan,
    Keypoint,
    PersonPose,
)

LANDMARK_COVERAGE_MARGIN = 1.6


def face_region(person: PersonPose) -> FaceRegion | None:
    landmarks = [
        person.keypoint(name) for name in FACE_LANDMARKS if person.keypoint(name).is_visible
    ]
    if not landmarks:
        return None

    centre_x = sum(landmark.x for landmark in landmarks) / len(landmarks)
    centre_y = sum(landmark.y for landmark in landmarks) / len(landmarks)
    radius = _radius(person, landmarks, centre_x, centre_y)
    if radius <= 0.0:
        return None
    return FaceRegion(centre_x=centre_x, centre_y=centre_y, radius=radius)


def _radius(
    person: PersonPose, landmarks: Sequence[Keypoint], centre_x: float, centre_y: float
) -> float:
    measured = max(_span_radius(person, span) for span in FACE_SPANS)
    if measured <= 0.0:
        measured = _span_radius(person, SHOULDER_SPAN)
    return max(measured, _landmark_radius(landmarks, centre_x, centre_y))


def _span_radius(person: PersonPose, span: FaceSpan) -> float:
    start = person.keypoint(span.start)
    end = person.keypoint(span.end)
    if not (start.is_visible and end.is_visible):
        return 0.0
    return math.hypot(start.x - end.x, start.y - end.y) * span.radius_scale


def _landmark_radius(landmarks: Sequence[Keypoint], centre_x: float, centre_y: float) -> float:
    spread = max(math.hypot(landmark.x - centre_x, landmark.y - centre_y) for landmark in landmarks)
    return spread * LANDMARK_COVERAGE_MARGIN
