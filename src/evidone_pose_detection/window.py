import logging

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent, QImage, QKeyEvent, QPixmap
from PySide6.QtWidgets import QLabel, QMainWindow, QSizePolicy

from evidone_pose_detection.camera import Camera
from evidone_pose_detection.models import Frame

logger = logging.getLogger(__name__)

WINDOW_TITLE = "Evidone Pose Detection"
QUIT_KEYS = frozenset({Qt.Key.Key_Q, Qt.Key.Key_Escape})


class VideoWindow(QMainWindow):
    def __init__(self, camera: Camera) -> None:
        super().__init__()
        properties = camera.properties()
        self._camera = camera
        self._view = _build_view()
        self._timer = QTimer(self)
        self._timer.setInterval(properties.frame_interval_ms)
        self._timer.timeout.connect(self._show_next_frame)

        self.setWindowTitle(WINDOW_TITLE)
        self.setCentralWidget(self._view)
        self.resize(properties.width, properties.height)

    def start(self) -> None:
        self._timer.start()

    def keyPressEvent(self, event: QKeyEvent, /) -> None:
        if event.key() in QUIT_KEYS:
            self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent, /) -> None:
        self._timer.stop()
        self._camera.release()
        super().closeEvent(event)

    def _show_next_frame(self) -> None:
        frame = self._camera.read()
        if frame is None:
            logger.warning("The camera stopped delivering frames; closing the window")
            self.close()
            return
        self._view.setPixmap(
            QPixmap.fromImage(_to_image(frame)).scaled(
                self._view.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


def _build_view() -> QLabel:
    view = QLabel()
    view.setAlignment(Qt.AlignmentFlag.AlignCenter)
    view.setMinimumSize(1, 1)
    view.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
    return view


def _to_image(frame: Frame) -> QImage:
    height, width, _ = frame.shape
    image = QImage(frame.data, width, height, frame.strides[0], QImage.Format.Format_BGR888)
    return image.copy()
