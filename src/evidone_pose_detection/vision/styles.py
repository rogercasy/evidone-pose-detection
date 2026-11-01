import cv2
import numpy as np

from evidone_pose_detection.models import Frame

# Every style reduces the face to a fixed number of samples across, whatever its size on
# screen, so a close-up loses as much detail as a distant face.
BLUR_SAMPLES = 6
PIXEL_BLOCKS = 8
FILL_COLOUR = (18, 18, 18)


def blurred(patch: Frame) -> Frame:
    height, width = patch.shape[:2]
    reduced = cv2.resize(patch, (BLUR_SAMPLES, BLUR_SAMPLES), interpolation=cv2.INTER_AREA)
    # Cubic, not linear: a linear upscale leaves visible facet edges between samples.
    return cv2.resize(reduced, (width, height), interpolation=cv2.INTER_CUBIC)


def pixelated(patch: Frame) -> Frame:
    height, width = patch.shape[:2]
    reduced = cv2.resize(patch, (PIXEL_BLOCKS, PIXEL_BLOCKS), interpolation=cv2.INTER_AREA)
    # Nearest keeps the block edges hard; the block size follows the region.
    return cv2.resize(reduced, (width, height), interpolation=cv2.INTER_NEAREST)


def filled(patch: Frame) -> Frame:
    return np.full(patch.shape, FILL_COLOUR, dtype=np.uint8)
