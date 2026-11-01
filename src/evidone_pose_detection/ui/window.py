import logging
import time
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QActionGroup, QCloseEvent, QImage, QKeyEvent, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QSizePolicy,
    QToolBar,
    QToolButton,
)

from evidone_pose_detection.camera import Camera, CaptureSession
from evidone_pose_detection.errors import CameraOpenError
from evidone_pose_detection.models import (
    DEFAULT_CAMERA_INDEX,
    CameraConfig,
    Frame,
    FrameStats,
    PersonPose,
    PoseSource,
)
from evidone_pose_detection.ui.framerate import FrameRate
from evidone_pose_detection.vision import blur_faces, draw_poses, draw_stats

logger = logging.getLogger(__name__)

WINDOW_TITLE = "Evidone Pose Detection"
QUIT_KEYS = frozenset({Qt.Key.Key_Q, Qt.Key.Key_Escape})
MINIMUM_YIELD_MS = 1

FRAME_RATE_WINDOW = 15

TOOLBAR_TITLE = "Main"
SETTINGS_LABEL = "Settings"
SOURCE_LABEL = "Source"
CAMERA_LABEL = "Camera"
VIDEO_LABEL = "Video File\u2026"
FILE_DIALOG_TITLE = "Choose a video file"
VIDEO_FILE_FILTER = "Video files (*.mp4 *.avi *.mov *.mkv *.webm);;All files (*)"
SOURCE_ERROR_TITLE = "Cannot open that source"

VIEW_BACKGROUND = "#101010"


class VideoWindow(QMainWindow):
    def __init__(self, camera: Camera, poses: PoseSource) -> None:
        super().__init__()
        properties = camera.properties()
        self._session = CaptureSession(camera)
        self._poses = poses
        self._people: tuple[PersonPose, ...] = ()
        self._frame_rate = FrameRate(FRAME_RATE_WINDOW)
        self._camera_index = _starting_camera_index(camera.config)
        self._view = _build_view()
        self._interval_ms = properties.frame_interval_ms
        self._running = False
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._show_next_frame)

        self.setWindowTitle(_title(0))
        self.setCentralWidget(self._view)
        self._build_toolbar()
        self._mark_active_source()
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
        self._session.release()
        super().closeEvent(event)

    def _show_next_frame(self) -> None:
        started = time.perf_counter()
        frame = self._session.camera.read()
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

    def _choose_camera(self) -> None:
        self._switch_to(CameraConfig(source=self._camera_index))

    def _choose_video_file(self) -> None:
        chosen = self._ask_for_video_file()
        if not chosen:
            self._mark_active_source()
            return
        self._switch_to(CameraConfig(source=Path(chosen)))

    def _switch_to(self, config: CameraConfig) -> None:
        if config == self._session.camera.config:
            self._mark_active_source()
            return
        try:
            camera = self._session.switch_to(config)
        except CameraOpenError as error:
            logger.warning("Keeping the running source: %s", error)
            self._report_source_error(error)
            self._mark_active_source()
            return

        if isinstance(config.source, int):
            self._camera_index = config.source
        self._interval_ms = camera.properties().frame_interval_ms
        self._mark_active_source()
        logger.info("Switched the source to %s", config.source)

    def _ask_for_video_file(self) -> str:
        chosen, _ = QFileDialog.getOpenFileName(self, FILE_DIALOG_TITLE, "", VIDEO_FILE_FILTER)
        return chosen

    def _report_source_error(self, error: CameraOpenError) -> None:
        QMessageBox.warning(self, SOURCE_ERROR_TITLE, str(error))

    def _mark_active_source(self) -> None:
        config = self._session.camera.config
        self._camera_action.setChecked(not config.is_file)
        self._video_action.setChecked(config.is_file)
        self._video_action.setText(_video_label(config))

    def _build_toolbar(self) -> None:
        toolbar = QToolBar(TOOLBAR_TITLE, self)
        toolbar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

        button = QToolButton(toolbar)
        button.setText(SETTINGS_LABEL)
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        button.setMenu(self._build_settings_menu())
        toolbar.addWidget(button)
        self._settings_button = button

    def _build_settings_menu(self) -> QMenu:
        menu = QMenu(SETTINGS_LABEL, self)
        self._source_menu = menu.addMenu(SOURCE_LABEL)

        group = QActionGroup(self)
        group.setExclusive(True)
        self._camera_action = self._source_menu.addAction(CAMERA_LABEL)
        self._video_action = self._source_menu.addAction(VIDEO_LABEL)
        for action in (self._camera_action, self._video_action):
            action.setCheckable(True)
            group.addAction(action)
        self._camera_action.triggered.connect(self._choose_camera)
        self._video_action.triggered.connect(self._choose_video_file)
        return menu

    def _show_people(self, people: tuple[PersonPose, ...]) -> None:
        if len(people) != len(self._people):
            self.setWindowTitle(_title(len(people)))
        self._people = people


def _starting_camera_index(config: CameraConfig) -> int:
    source = config.source
    return source if isinstance(source, int) else DEFAULT_CAMERA_INDEX


def _video_label(config: CameraConfig) -> str:
    source = config.source
    if isinstance(source, Path):
        return f"{VIDEO_LABEL} ({source.name})"
    return VIDEO_LABEL


def _title(people: int) -> str:
    return f"{WINDOW_TITLE} — {people} detected"


def _build_view() -> QLabel:
    view = QLabel()
    view.setStyleSheet(f"background-color: {VIEW_BACKGROUND};")
    view.setAlignment(Qt.AlignmentFlag.AlignCenter)
    view.setMinimumSize(1, 1)
    view.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
    return view


def _to_image(frame: Frame) -> QImage:
    height, width, _ = frame.shape
    image = QImage(frame.data, width, height, frame.strides[0], QImage.Format.Format_BGR888)
    return image.copy()
