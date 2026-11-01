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
        self._replays = 0

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

    @property
    def config(self) -> CameraConfig:
        return self._config

    @property
    def replays(self) -> int:
        return self._replays

    def open(self) -> None:
        source = self._config.source
        try:
            capture = cv2.VideoCapture(source if isinstance(source, int) else str(source))
        except Exception as error:
            raise CameraOpenError(source) from error

        if not capture.isOpened():
            capture.release()
            raise CameraOpenError(source)
        self._capture = capture
        logger.info("Opened capture source %s", source)

    def properties(self) -> CaptureProperties:
        capture = self._open_capture()
        fps = capture.get(cv2.CAP_PROP_FPS)
        return CaptureProperties(
            width=round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=fps if fps > 0 else FALLBACK_FPS,
        )

    def read(self) -> Frame | None:
        capture = self._open_capture()
        read, frame = capture.read()
        if read:
            return frame
        if not self._config.is_file:
            return None
        return self._replay(capture)

    def _replay(self, capture: cv2.VideoCapture) -> Frame | None:
        # A clip that ran out starts again; a clip that yields nothing twice is finished.
        logger.debug("Rewinding %s", self._config.source)
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        read, frame = capture.read()
        if not read:
            return None
        self._replays += 1
        return frame

    def release(self) -> None:
        if self._capture is None:
            return
        self._capture.release()
        self._capture = None
        logger.info("Released capture source %s", self._config.source)

    def _open_capture(self) -> cv2.VideoCapture:
        if self._capture is None:
            raise CameraNotOpenError
        return self._capture


class CaptureSession:
    def __init__(self, camera: Camera) -> None:
        self._camera = camera

    @property
    def camera(self) -> Camera:
        return self._camera

    def switch_to(self, config: CameraConfig) -> Camera:
        replacement = Camera(config)
        replacement.open()
        # Only once the replacement is open does the previous source let go, so a
        # failure leaves the running source untouched.
        self._camera.release()
        self._camera = replacement
        return replacement

    def release(self) -> None:
        self._camera.release()
