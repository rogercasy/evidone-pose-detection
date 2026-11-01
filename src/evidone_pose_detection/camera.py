import logging
from types import TracebackType

import cv2

from evidone_pose_detection.errors import CameraNotOpenError, CameraOpenError
from evidone_pose_detection.models import CameraConfig, CaptureProperties, Frame

logger = logging.getLogger(__name__)

FALLBACK_FPS = 30.0


class Camera:
    def __init__(self, config: CameraConfig) -> None:
        self._config = config
        self._capture: cv2.VideoCapture | None = None

    def __enter__(self) -> "Camera":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

    def open(self) -> None:
        try:
            capture = cv2.VideoCapture(self._config.device_index)
        except Exception as error:
            raise CameraOpenError(self._config.device_index) from error

        if not capture.isOpened():
            capture.release()
            raise CameraOpenError(self._config.device_index)
        self._capture = capture
        logger.info("Opened camera device %d", self._config.device_index)

    def properties(self) -> CaptureProperties:
        capture = self._opened_capture()
        fps = capture.get(cv2.CAP_PROP_FPS)
        return CaptureProperties(
            width=round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=fps if fps > 0 else FALLBACK_FPS,
        )

    def read(self) -> Frame | None:
        read, frame = self._opened_capture().read()
        return frame if read else None

    def release(self) -> None:
        if self._capture is None:
            return
        self._capture.release()
        self._capture = None
        logger.info("Released camera device %d", self._config.device_index)

    def _opened_capture(self) -> cv2.VideoCapture:
        if self._capture is None:
            raise CameraNotOpenError
        return self._capture
