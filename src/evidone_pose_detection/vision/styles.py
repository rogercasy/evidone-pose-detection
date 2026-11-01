import cv2
import numpy as np

from evidone_pose_detection.config import CONFIG
from evidone_pose_detection.models import Frame


def blurred(patch: Frame) -> Frame:
    height, width = patch.shape[:2]
    reduced = cv2.resize(
        patch,
        (CONFIG.hiding.blur_samples, CONFIG.hiding.blur_samples),
        interpolation=cv2.INTER_AREA,
    )
    # Cubic, not linear: a linear upscale leaves visible facet edges between samples.
    return cv2.resize(reduced, (width, height), interpolation=cv2.INTER_CUBIC)


def pixelated(patch: Frame) -> Frame:
    height, width = patch.shape[:2]
    reduced = cv2.resize(
        patch,
        (CONFIG.hiding.pixel_blocks, CONFIG.hiding.pixel_blocks),
        interpolation=cv2.INTER_AREA,
    )
    # Nearest keeps the block edges hard; the block size follows the region.
    return cv2.resize(reduced, (width, height), interpolation=cv2.INTER_NEAREST)


def filled(patch: Frame) -> Frame:
    return np.full(patch.shape, CONFIG.hiding.fill_colour, dtype=np.uint8)
