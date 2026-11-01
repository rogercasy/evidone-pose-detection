from collections.abc import Callable, Mapping, Sequence
from math import ceil, floor
from types import MappingProxyType

import cv2
import numpy as np

from evidone_pose_detection.models import FaceRegion, Frame, HideStyle
from evidone_pose_detection.vision.styles import blurred, filled, pixelated

type Conceal = Callable[[Frame], Frame]

STYLES: Mapping[HideStyle, Conceal] = MappingProxyType(
    {HideStyle.BLUR: blurred, HideStyle.PIXELATE: pixelated, HideStyle.FILL: filled}
)


def hide_faces(frame: Frame, regions: Sequence[FaceRegion], style: HideStyle) -> Frame:
    conceal = STYLES[style]
    for region in regions:
        _hide(frame, region, conceal)
    return frame


def _hide(frame: Frame, region: FaceRegion, conceal: Conceal) -> None:
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
    patch[mask > 0] = conceal(patch)[mask > 0]


def _clipped_bounds(frame: Frame, region: FaceRegion) -> tuple[int, int, int, int] | None:
    height, width = frame.shape[:2]
    left = max(0, floor(region.centre_x - region.radius))
    top = max(0, floor(region.centre_y - region.radius))
    right = min(width, ceil(region.centre_x + region.radius))
    bottom = min(height, ceil(region.centre_y + region.radius))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom
