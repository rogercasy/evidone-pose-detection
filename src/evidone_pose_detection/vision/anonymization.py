from collections.abc import Sequence
from math import ceil, floor

import cv2
import numpy as np

from evidone_pose_detection.models import FaceRegion, Frame, PersonPose
from evidone_pose_detection.vision.faces import face_region

FACE_SAMPLE_GRID = 6


def blur_faces(frame: Frame, people: Sequence[PersonPose]) -> Frame:
    for person in people:
        region = face_region(person)
        if region is not None:
            _hide(frame, region)
    return frame


def _hide(frame: Frame, region: FaceRegion) -> None:
    bounds = _clipped_bounds(frame, region)
    if bounds is None:
        return

    left, top, right, bottom = bounds
    patch = frame[top:bottom, left:right]
    mask = np.zeros(patch.shape[:2], dtype=np.uint8)
    cv2.circle(
        mask,
        (round(region.centre_x) - left, round(region.centre_y) - top),
        round(region.radius),
        255,
        cv2.FILLED,
    )
    patch[mask > 0] = _blurred(patch)[mask > 0]


def _clipped_bounds(frame: Frame, region: FaceRegion) -> tuple[int, int, int, int] | None:
    height, width = frame.shape[:2]
    left = max(0, floor(region.centre_x - region.radius))
    top = max(0, floor(region.centre_y - region.radius))
    right = min(width, ceil(region.centre_x + region.radius))
    bottom = min(height, ceil(region.centre_y + region.radius))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _blurred(patch: Frame) -> Frame:
    height, width = patch.shape[:2]
    reduced = cv2.resize(patch, (FACE_SAMPLE_GRID, FACE_SAMPLE_GRID), interpolation=cv2.INTER_AREA)
    # Cubic, not linear: a linear upscale leaves visible facet edges between samples.
    return cv2.resize(reduced, (width, height), interpolation=cv2.INTER_CUBIC)
