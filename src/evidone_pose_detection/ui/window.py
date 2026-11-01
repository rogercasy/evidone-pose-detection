import logging
import time
from collections.abc import Callable
from datetime import UTC, datetime
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
from evidone_pose_detection.config import CONFIG
from evidone_pose_detection.detection_log import DetectionLog
from evidone_pose_detection.errors import (
    CameraOpenError,
    DetectionLogError,
    EvidoneError,
    RecordingError,
)
from evidone_pose_detection.models import (
    CameraConfig,
    Frame,
    FrameStats,
    HideStyle,
    PersonPose,
    PoseSource,
    ViewSettings,
)
from evidone_pose_detection.recording import Recorder
from evidone_pose_detection.ui.framerate import FrameRate
from evidone_pose_detection.vision import (
    FaceTracker,
    draw_poses,
    draw_stats,
    face_regions,
    hide_faces,
)

logger = logging.getLogger(__name__)

QUIT_KEYS = frozenset({Qt.Key.Key_Q, Qt.Key.Key_Escape})


class VideoWindow(QMainWindow):
    def __init__(self, camera: Camera, poses: PoseSource) -> None:
        super().__init__()
        properties = camera.properties()
        self._session = CaptureSession(camera)
        self._poses = poses
        self._people: tuple[PersonPose, ...] = ()
        self._frame_rate = FrameRate(CONFIG.window.frame_rate_window)
        self._camera_index = _starting_camera_index(camera.config)
        self._tracker = FaceTracker(hold_frames=CONFIG.tracking.hold_frames)
        self._settings = ViewSettings(
            hide_style=CONFIG.hiding.default_style, draw_skeleton=CONFIG.skeleton.draw_by_default
        )
        self._recorder = Recorder()
        self._detection_log = DetectionLog()
        self._replays = camera.replays
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
        self._mark_active_view()
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
        self.shutdown()
        self._session.release()
        super().closeEvent(event)

    def shutdown(self) -> None:
        self._running = False
        self._timer.stop()
        self._recorder.close()
        self._detection_log.close()

    def _show_next_frame(self) -> None:
        started = time.perf_counter()
        frame = self._session.camera.read()
        if frame is None:
            logger.warning("The camera stopped delivering frames; closing the window")
            self.close()
            return

        self._forget_tracks_across_a_replay()
        result = self._poses.estimate(frame)
        self._detection_log.write(result.people)
        self._show_people(result.people)
        hide_faces(
            frame, self._tracker.track(face_regions(result.people)), self._settings.hide_style
        )
        if self._settings.draw_skeleton:
            draw_poses(frame, result.people)
        draw_stats(
            frame,
            FrameStats(
                inference_ms=result.timing.inference_ms,
                frames_per_second=self._frame_rate.tick(time.perf_counter()),
                people=len(result.people),
            ),
        )
        self._record(frame)
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
        self._timer.start(max(CONFIG.window.minimum_yield_ms, self._interval_ms - elapsed_ms))

    def _forget_tracks_across_a_replay(self) -> None:
        replays = self._session.camera.replays
        if replays != self._replays:
            self._replays = replays
            self._tracker.reset()

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
            self._report(CONFIG.labels.source_error_title, error)
            self._mark_active_source()
            return

        if isinstance(config.source, int):
            self._camera_index = config.source
        self._interval_ms = camera.properties().frame_interval_ms
        self._tracker.reset()
        self._replays = camera.replays
        self._mark_active_source()
        logger.info("Switched the source to %s", config.source)

    def _ask_for_video_file(self) -> str:
        chosen, _ = QFileDialog.getOpenFileName(
            self, CONFIG.labels.file_dialog_title, "", CONFIG.labels.video_file_filter
        )
        return chosen

    def _report(self, title: str, error: EvidoneError) -> None:
        QMessageBox.warning(self, title, str(error))

    def _style_chooser(self, style: HideStyle) -> Callable[[], None]:
        def choose() -> None:
            self._settings.hide_style = style
            self._mark_active_view()

        return choose

    def _set_recording(self, recording: bool) -> None:
        if recording:
            self._start_recording()
            return
        self._stop_recording()

    def _start_recording(self) -> None:
        self._recorder.start(_recording_path(), _frames_per_second(self._interval_ms))
        self._mark_active_view()

    def _stop_recording(self) -> None:
        self._recorder.close()
        self._mark_active_view()

    def _set_logging(self, logging_on: bool) -> None:
        if logging_on:
            self._start_logging()
            return
        self._stop_logging()

    def _start_logging(self) -> None:
        try:
            self._detection_log.start(_detection_log_path())
        except DetectionLogError as error:
            logger.warning("Detection logging did not start: %s", error)
            self._report(CONFIG.labels.log_error_title, error)
        self._mark_active_view()

    def _stop_logging(self) -> None:
        self._detection_log.close()
        self._mark_active_view()

    def _record(self, frame: Frame) -> None:
        try:
            self._recorder.write(frame)
        except RecordingError as error:
            logger.warning("Recording stopped: %s", error)
            self._stop_recording()
            self._report(CONFIG.labels.recording_error_title, error)

    def _toggle_skeleton(self, drawn: bool) -> None:
        self._settings.draw_skeleton = drawn
        self._mark_active_view()

    def _mark_active_view(self) -> None:
        for style, action in self._style_actions.items():
            action.setChecked(style is self._settings.hide_style)
        self._skeleton_action.setChecked(self._settings.draw_skeleton)
        self._record_action.setChecked(self._recorder.is_recording)
        self._log_action.setChecked(self._detection_log.is_logging)

    def _mark_active_source(self) -> None:
        config = self._session.camera.config
        self._camera_action.setChecked(not config.is_file)
        self._video_action.setChecked(config.is_file)
        self._video_action.setText(_video_label(config))

    def _build_toolbar(self) -> None:
        toolbar = QToolBar(CONFIG.labels.toolbar, self)
        toolbar.setMovable(False)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, toolbar)

        button = QToolButton(toolbar)
        button.setText(CONFIG.labels.settings)
        button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        button.setMenu(self._build_settings_menu())
        toolbar.addWidget(button)
        self._settings_button = button

    def _build_settings_menu(self) -> QMenu:
        menu = QMenu(CONFIG.labels.settings, self)
        self._source_menu = menu.addMenu(CONFIG.labels.source)

        group = QActionGroup(self)
        group.setExclusive(True)
        self._camera_action = self._source_menu.addAction(CONFIG.labels.camera)
        self._video_action = self._source_menu.addAction(CONFIG.labels.video_file)
        for action in (self._camera_action, self._video_action):
            action.setCheckable(True)
            group.addAction(action)
        self._camera_action.triggered.connect(self._choose_camera)
        self._video_action.triggered.connect(self._choose_video_file)

        self._build_hiding_menu(menu)
        menu.addSeparator()
        self._skeleton_action = menu.addAction(CONFIG.labels.skeleton)
        self._skeleton_action.setCheckable(True)
        self._skeleton_action.toggled.connect(self._toggle_skeleton)

        menu.addSeparator()
        self._record_action = menu.addAction(CONFIG.labels.record)
        self._record_action.setCheckable(True)
        self._record_action.toggled.connect(self._set_recording)

        self._log_action = menu.addAction(CONFIG.labels.detection_log)
        self._log_action.setCheckable(True)
        self._log_action.toggled.connect(self._set_logging)
        return menu

    def _build_hiding_menu(self, menu: QMenu) -> None:
        self._hiding_menu = menu.addMenu(CONFIG.labels.hiding)
        styles = QActionGroup(self)
        styles.setExclusive(True)
        self._style_actions = {}
        for style in HideStyle:
            action = self._hiding_menu.addAction(CONFIG.labels.styles[style])
            action.setCheckable(True)
            styles.addAction(action)
            action.triggered.connect(self._style_chooser(style))
            self._style_actions[style] = action

    def _show_people(self, people: tuple[PersonPose, ...]) -> None:
        if len(people) != len(self._people):
            self.setWindowTitle(_title(len(people)))
        self._people = people


def _recording_path() -> Path:
    return _output_path(".mp4")


def _detection_log_path() -> Path:
    return _output_path(".jsonl")


def _output_path(suffix: str) -> Path:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
    return Path(f"{CONFIG.output.stem}-{stamp}{suffix}")


def _frames_per_second(interval_ms: int) -> float:
    return 1000.0 / interval_ms


def _starting_camera_index(config: CameraConfig) -> int:
    source = config.source
    return source if isinstance(source, int) else CONFIG.camera.default_index


def _video_label(config: CameraConfig) -> str:
    source = config.source
    if isinstance(source, Path):
        return f"{CONFIG.labels.video_file} ({source.name})"
    return CONFIG.labels.video_file


def _title(people: int) -> str:
    return f"{CONFIG.window.title} — {people} detected"


def _build_view() -> QLabel:
    view = QLabel()
    view.setStyleSheet(f"background-color: {CONFIG.window.view_background};")
    view.setAlignment(Qt.AlignmentFlag.AlignCenter)
    view.setMinimumSize(1, 1)
    view.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
    return view


def _to_image(frame: Frame) -> QImage:
    height, width, _ = frame.shape
    image = QImage(frame.data, width, height, frame.strides[0], QImage.Format.Format_BGR888)
    return image.copy()
