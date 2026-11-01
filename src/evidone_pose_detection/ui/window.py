import logging
import time

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QCloseEvent, QImage, QKeyEvent, QPixmap
from PySide6.QtWidgets import QLabel, QMainWindow, QSizePolicy

from evidone_pose_detection.camera import Camera
from evidone_pose_detection.models import Frame, FrameStats, PersonPose, PoseSource
from evidone_pose_detection.ui.framerate import FrameRate
from evidone_pose_detection.vision import blur_faces, draw_poses, draw_stats

logger = logging.getLogger(__name__)

WINDOW_TITLE = "Evidone Pose Detection"
QUIT_KEYS = frozenset({Qt.Key.Key_Q, Qt.Key.Key_Escape})
MINIMUM_YIELD_MS = 1

# Frames averaged for the displayed rate, so the number is steady enough to read.
FRAME_RATE_WINDOW = 15


class VideoWindow(QMainWindow):
    def __init__(self, camera: Camera, poses: PoseSource) -> None:
        super().__init__()
        properties = camera.properties()
        self._camera = camera
        self._poses = poses
        self._people: tuple[PersonPose, ...] = ()
        self._frame_rate = FrameRate(FRAME_RATE_WINDOW)
        self._view = _build_view()
        self._interval_ms = properties.frame_interval_ms
        self._running = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._show_next_frame)

        self.setWindowTitle(_title(0))
        self.setCentralWidget(self._view)
        self.resize(properties.width, properties.height)

    def start(self) -> None:
        self._running = True
        self._timer.start(self._interval_ms)

    def keyPressEvent(self, event: QKeyEvent, /) -> None:
        if event.key() in QUIT_KEYS:
            self.close()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent, /) -> None:
        self._running = False
        self._timer.stop()
        self._camera.release()
        super().closeEvent(event)

    def _show_next_frame(self) -> None:
        started = time.perf_counter()
        frame = self._camera.read()
        if frame is None:
            logger.warning("The camera stopped delivering frames; closing the window")
            self.close()
            return

        result = self._poses.estimate(frame)
        self._show_people(result.people)
        blur_faces(frame, result.people)
        draw_poses(frame, result.people)
        draw_stats(
            frame,
            FrameStats(
                inference_ms=result.timing.inference_ms,
                frames_per_second=self._frame_rate.tick(time.perf_counter()),
                people=len(result.people),
            ),
        )
        self._view.setPixmap(
            QPixmap.fromImage(_to_image(frame)).scaled(
                self._view.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        self._schedule_next_frame(started)

    def _schedule_next_frame(self, started: float) -> None:
        if not self._running:
            return
        elapsed_ms = round((time.perf_counter() - started) * 1000)
        self._timer.start(max(MINIMUM_YIELD_MS, self._interval_ms - elapsed_ms))

    def _show_people(self, people: tuple[PersonPose, ...]) -> None:
        if len(people) != len(self._people):
            self.setWindowTitle(_title(len(people)))
        self._people = people


def _title(people: int) -> str:
    return f"{WINDOW_TITLE} — {people} detected"


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
