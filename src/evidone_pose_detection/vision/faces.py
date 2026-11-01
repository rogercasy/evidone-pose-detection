import math
from collections.abc import Sequence

from evidone_pose_detection.config import CONFIG
from evidone_pose_detection.models import (
    FACE_LANDMARKS,
    COCOKeypoint,
    FaceRegion,
    FaceSpan,
    Keypoint,
    PersonPose,
)
from evidone_pose_detection.vision.visibility import is_visible


def face_regions(people: Sequence[PersonPose]) -> tuple[FaceRegion, ...]:
    found = (face_region(person) for person in people)
    return tuple(region for region in found if region is not None)


def face_region(person: PersonPose) -> FaceRegion | None:
    landmarks = [
        person.keypoint(name) for name in FACE_LANDMARKS if is_visible(person.keypoint(name))
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
    measured = max(_span_radius(person, span) for span in CONFIG.faces.spans)
    if measured <= 0.0:
        measured = _span_radius(person, _shoulder_span())
    return max(measured, _landmark_radius(landmarks, centre_x, centre_y))


def _span_radius(person: PersonPose, span: FaceSpan) -> float:
    start = person.keypoint(span.start)
    end = person.keypoint(span.end)
    if not (is_visible(start) and is_visible(end)):
        return 0.0
    return math.hypot(start.x - end.x, start.y - end.y) * span.radius_scale


def _landmark_radius(landmarks: Sequence[Keypoint], centre_x: float, centre_y: float) -> float:
    spread = max(math.hypot(landmark.x - centre_x, landmark.y - centre_y) for landmark in landmarks)
    return spread * CONFIG.faces.landmark_coverage_margin


def _shoulder_span() -> FaceSpan:
    return FaceSpan(
        start=COCOKeypoint.LEFT_SHOULDER,
        end=COCOKeypoint.RIGHT_SHOULDER,
        radius_scale=CONFIG.faces.shoulder_scale,
    )
