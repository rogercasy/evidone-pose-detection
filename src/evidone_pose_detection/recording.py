import logging
from pathlib import Path

import cv2
import numpy as np

from evidone_pose_detection.errors import RecordingError
from evidone_pose_detection.models import Frame

logger = logging.getLogger(__name__)

FOURCC = "mp4v"


class Recorder:
    def __init__(self) -> None:
        self._writer: cv2.VideoWriter | None = None
        self._destination: Path | None = None
        self._size: tuple[int, int] | None = None
        self._fps = 0.0

    @property
    def is_recording(self) -> bool:
        return self._destination is not None

    @property
    def destination(self) -> Path | None:
        return self._destination

    def start(self, destination: Path, fps: float) -> None:
        self.close()
        self._destination = destination
        self._fps = fps
        logger.info("Recording to %s", destination)

    def write(self, frame: Frame) -> None:
        if self._destination is None:
            return
        if self._writer is None:
            self._open(frame)
        if self._writer is not None and self._size is not None:
            self._writer.write(_fitted(frame, self._size))

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            logger.info("Finished recording %s", self._destination)
        self._writer = None
        self._destination = None
        self._size = None

    def _open(self, frame: Frame) -> None:
        # The size is only known once a frame has arrived, so the file opens here.
        destination = self._destination
        if destination is None:
            return

        height, width = frame.shape[:2]
        writer = cv2.VideoWriter(
            str(destination), cv2.VideoWriter.fourcc(*FOURCC), self._fps, (width, height)
        )
        if not writer.isOpened():
            writer.release()
            self._destination = None
            raise RecordingError(destination)
        self._writer = writer
        self._size = (width, height)


def _fitted(frame: Frame, size: tuple[int, int]) -> Frame:
    # A later source may have a different resolution; an mp4 cannot change size part
    # way through, so the frame is letterboxed into the size the file was opened with.
    width, height = size
    source_height, source_width = frame.shape[:2]
    if (source_width, source_height) == size:
        return frame

    scale = min(width / source_width, height / source_height)
    scaled_width = max(1, round(source_width * scale))
    scaled_height = max(1, round(source_height * scale))
    scaled = cv2.resize(frame, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    left = (width - scaled_width) // 2
    top = (height - scaled_height) // 2
    canvas[top : top + scaled_height, left : left + scaled_width] = scaled
    return canvas
