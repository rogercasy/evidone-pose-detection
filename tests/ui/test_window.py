import json
from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QToolBar, QToolButton

from conftest import FRAME_HEIGHT, FRAME_WIDTH, FakeCapture, FakePoseSource, make_person
from evidone_pose_detection.camera import Camera
from evidone_pose_detection.config import CONFIG
from evidone_pose_detection.models import CameraConfig, Frame, HideStyle, PersonPose, PoseResult
from evidone_pose_detection.ui.window import VideoWindow

InstallCapture = Callable[[FakeCapture], list[int]]


@pytest.fixture
def window(
    application: QApplication,
    working_capture: FakeCapture,
    install_capture: InstallCapture,
    no_poses: FakePoseSource,
) -> VideoWindow:
    install_capture(working_capture)
    return VideoWindow(Camera(CameraConfig(source=0)).__enter__(), no_poses)


def test_window_adopts_the_title_and_the_frame_size(window: VideoWindow) -> None:
    assert window.windowTitle().startswith(CONFIG.window.title)
    assert (window.width(), window.height()) == (FRAME_WIDTH, FRAME_HEIGHT)


def test_every_captured_frame_is_sent_to_the_pose_model(
    window: VideoWindow, no_poses: FakePoseSource
) -> None:
    for _ in range(3):
        window._show_next_frame()

    assert no_poses.seen_frames == 3


def test_a_frame_the_camera_never_delivered_is_not_sent_to_the_model(
    application: QApplication, install_capture: InstallCapture, no_poses: FakePoseSource
) -> None:
    install_capture(FakeCapture(frames=[], opened=True, fps=25.0))
    window = VideoWindow(Camera(CameraConfig(source=0)).__enter__(), no_poses)

    window._show_next_frame()

    assert no_poses.seen_frames == 0


@pytest.mark.parametrize("people", [0, 1, 4])
def test_the_title_reports_how_many_people_were_detected(
    application: QApplication,
    working_capture: FakeCapture,
    install_capture: InstallCapture,
    people: int,
) -> None:
    install_capture(working_capture)
    poses = FakePoseSource(tuple(make_person(0.8) for _ in range(people)))
    window = VideoWindow(Camera(CameraConfig(source=0)).__enter__(), poses)

    window._show_next_frame()

    assert window.windowTitle() == f"{CONFIG.window.title} \u2014 {people} detected"


def test_start_schedules_a_single_frame_at_the_capture_rate(window: VideoWindow) -> None:
    window.start()

    assert window._timer.isActive()
    assert window._timer.isSingleShot()
    assert window._timer.interval() == 40


def test_each_frame_rearms_the_next_one(window: VideoWindow) -> None:
    window.start()

    window._show_next_frame()

    assert window._timer.isActive()
    assert CONFIG.window.minimum_yield_ms <= window._timer.interval() <= 40


def test_a_frame_slower_than_the_capture_rate_still_yields_to_the_event_loop(
    window: VideoWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    window.start()
    calls = iter(range(1000))

    def clock() -> float:
        # Zero when the frame starts, ten seconds later for every reading after that.
        return 0.0 if next(calls) == 0 else 10.0

    monkeypatch.setattr("evidone_pose_detection.ui.window.time.perf_counter", clock)

    window._show_next_frame()

    assert window._timer.interval() == CONFIG.window.minimum_yield_ms


def test_a_closed_window_does_not_rearm_the_frame_loop(window: VideoWindow) -> None:
    window.start()
    window.close()

    window._schedule_next_frame(started=0.0)

    assert not window._timer.isActive()


def test_a_captured_frame_fills_the_view_without_distortion(window: VideoWindow) -> None:
    window.show()

    window._show_next_frame()

    pixmap = window._view.pixmap()
    assert not pixmap.isNull()
    assert pixmap.width() <= window._view.width()
    assert pixmap.height() <= window._view.height()
    assert pixmap.width() / pixmap.height() == pytest.approx(FRAME_WIDTH / FRAME_HEIGHT, abs=0.05)


@pytest.mark.parametrize("key", [Qt.Key.Key_Q, Qt.Key.Key_Escape])
def test_quit_keys_close_the_window_and_release_the_device(
    window: VideoWindow, working_capture: FakeCapture, key: Qt.Key
) -> None:
    window.show()
    window.start()

    window.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier))

    assert not window.isVisible()
    assert not window._timer.isActive()
    assert working_capture.released == 1


@pytest.mark.parametrize("key", [Qt.Key.Key_A, Qt.Key.Key_Space])
def test_other_keys_leave_the_window_running(
    window: VideoWindow, working_capture: FakeCapture, key: Qt.Key
) -> None:
    window.show()

    window.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier))

    assert window.isVisible()
    assert working_capture.released == 0


def test_a_device_that_stops_delivering_closes_the_window(
    application: QApplication, install_capture: InstallCapture
) -> None:
    capture = FakeCapture(frames=[], opened=True, fps=25.0)
    install_capture(capture)
    window = VideoWindow(Camera(CameraConfig(source=0)).__enter__(), FakePoseSource(()))
    window.show()
    window.start()

    window._show_next_frame()

    assert not window.isVisible()
    assert capture.released == 1


def test_the_frame_shown_in_the_window_carries_the_drawn_skeleton(
    application: QApplication, working_capture: FakeCapture, install_capture: InstallCapture
) -> None:
    install_capture(working_capture)
    poses = FakePoseSource((make_person(0.8),))
    window = VideoWindow(Camera(CameraConfig(source=0)).__enter__(), poses)

    window._show_next_frame()

    assert (working_capture.delivered[-1] != 128).any()


def test_the_model_never_sees_a_frame_we_have_drawn_on(
    application: QApplication, working_capture: FakeCapture, install_capture: InstallCapture
) -> None:
    install_capture(working_capture)
    poses = FakePoseSource((make_person(0.8),))
    window = VideoWindow(Camera(CameraConfig(source=0)).__enter__(), poses)

    window._show_next_frame()
    window._show_next_frame()

    assert poses.last_frame is not None
    assert (poses.last_frame == 128).all()


def test_faces_are_hidden_before_the_skeleton_is_drawn(
    application: QApplication,
    working_capture: FakeCapture,
    install_capture: InstallCapture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_capture(working_capture)
    order: list[str] = []

    class RecordingPoses(FakePoseSource):
        def estimate(self, frame: Frame) -> PoseResult:
            order.append("estimate")
            return super().estimate(frame)

    def record_hide(frame: Frame, regions: Sequence[object], style: object) -> Frame:
        order.append("hide")
        return frame

    def record_draw(frame: Frame, people: Sequence[PersonPose]) -> Frame:
        order.append("draw")
        return frame

    monkeypatch.setattr("evidone_pose_detection.ui.window.hide_faces", record_hide)
    monkeypatch.setattr("evidone_pose_detection.ui.window.draw_poses", record_draw)
    window = VideoWindow(
        Camera(CameraConfig(source=0)).__enter__(), RecordingPoses((make_person(0.8),))
    )

    window._show_next_frame()

    assert order == ["estimate", "hide", "draw"]


def test_the_skeleton_stays_sharp_on_top_of_a_blurred_face(
    application: QApplication, working_capture: FakeCapture, install_capture: InstallCapture
) -> None:
    install_capture(working_capture)
    poses = FakePoseSource((make_person(0.8),))
    window = VideoWindow(Camera(CameraConfig(source=0)).__enter__(), poses)

    window._show_next_frame()

    shown = working_capture.delivered[-1]
    drawn_colours = {tuple(int(c) for c in colour) for colour in CONFIG.skeleton.colours.values()}
    painted = {tuple(int(c) for c in pixel) for row in shown for pixel in row}
    assert drawn_colours & painted, "no un-smeared skeleton colour survived the blur"


# --- toolbar and live source switching -------------------------------------------------

InstallCaptures = Callable[[list[FakeCapture]], list[object]]
CLIP = Path("clip.avi")


def window_over(captures: list[FakeCapture], install: InstallCaptures) -> VideoWindow:
    install(captures)
    return VideoWindow(Camera(CameraConfig(source=0)).__enter__(), FakePoseSource(()))


def choose_file(window: VideoWindow, chosen: str) -> list[str]:
    reported: list[str] = []
    window._ask_for_video_file = lambda: chosen  # type: ignore[method-assign]
    window._report = lambda title, error: reported.append(str(error))  # type: ignore[method-assign]
    window._video_action.trigger()
    return reported


def running(frame: Frame) -> FakeCapture:
    return FakeCapture(frames=[frame] * 8, opened=True, fps=25.0)


def test_the_window_has_a_toolbar_with_a_settings_button(
    application: QApplication, frame: Frame, install_captures: InstallCaptures
) -> None:
    window = window_over([running(frame)], install_captures)

    toolbars = window.findChildren(QToolBar)
    buttons = window.findChildren(QToolButton)

    assert len(toolbars) == 1
    assert window.toolBarArea(toolbars[0]) == Qt.ToolBarArea.TopToolBarArea
    assert any(button.text() == CONFIG.labels.settings for button in buttons)


def test_the_settings_menu_offers_a_camera_and_a_video_file(
    application: QApplication, frame: Frame, install_captures: InstallCaptures
) -> None:
    window = window_over([running(frame)], install_captures)

    assert window._source_menu.title() == CONFIG.labels.source
    labels = [action.text() for action in window._source_menu.actions()]
    assert labels[0] == CONFIG.labels.camera
    assert labels[1].startswith(CONFIG.labels.video_file)


def test_the_menu_marks_the_camera_as_the_active_source_at_startup(
    application: QApplication, frame: Frame, install_captures: InstallCaptures
) -> None:
    window = window_over([running(frame)], install_captures)

    assert window._camera_action.isChecked()
    assert not window._video_action.isChecked()


def test_choosing_a_video_file_swaps_the_source_and_releases_the_camera(
    application: QApplication, frame: Frame, install_captures: InstallCaptures
) -> None:
    camera, clip = running(frame), running(frame)
    window = window_over([camera, clip], install_captures)

    reported = choose_file(window, str(CLIP))

    assert reported == []
    assert camera.released == 1, "the camera was left open behind the clip"
    assert clip.released == 0
    assert window._session.camera.config == CameraConfig(source=CLIP)


def test_the_menu_names_the_active_video_file(
    application: QApplication, frame: Frame, install_captures: InstallCaptures
) -> None:
    window = window_over([running(frame), running(frame)], install_captures)

    choose_file(window, str(CLIP))

    assert window._video_action.isChecked()
    assert not window._camera_action.isChecked()
    assert CLIP.name in window._video_action.text()


def test_switching_back_to_the_camera_releases_the_clip(
    application: QApplication, frame: Frame, install_captures: InstallCaptures
) -> None:
    camera, clip, again = running(frame), running(frame), running(frame)
    window = window_over([camera, clip, again], install_captures)
    choose_file(window, str(CLIP))

    window._camera_action.trigger()

    assert clip.released == 1, "the clip handle was left open"
    assert window._camera_action.isChecked()
    assert window._session.camera.config == CameraConfig(source=0)


def test_cancelling_the_dialog_leaves_everything_as_it_was(
    application: QApplication, frame: Frame, install_captures: InstallCaptures
) -> None:
    camera = running(frame)
    window = window_over([camera], install_captures)
    before = window._session.camera

    reported = choose_file(window, "")

    assert reported == []
    assert window._session.camera is before
    assert camera.released == 0
    assert window._camera_action.isChecked()
    assert not window._video_action.isChecked()
    assert window._video_action.text() == CONFIG.labels.video_file


def test_a_file_that_cannot_be_opened_is_reported_and_the_source_keeps_running(
    application: QApplication, frame: Frame, install_captures: InstallCaptures
) -> None:
    camera = running(frame)
    broken = FakeCapture(frames=[], opened=False, fps=25.0)
    window = window_over([camera, broken], install_captures)

    reported = choose_file(window, "missing.avi")

    assert len(reported) == 1
    assert "missing.avi" in reported[0]
    assert camera.released == 0, "a bad path tore down the running source"
    assert window._session.camera.read() is not None, "the source stopped delivering"
    assert window._camera_action.isChecked()
    assert not window._video_action.isChecked()


def test_a_failed_switch_does_not_close_the_window(
    application: QApplication, frame: Frame, install_captures: InstallCaptures
) -> None:
    window = window_over(
        [running(frame), FakeCapture([], opened=False, fps=25.0)], install_captures
    )
    window.show()

    choose_file(window, "missing.avi")

    assert window.isVisible(), "a mistyped path closed the session"


def test_the_frame_loop_keeps_running_on_the_new_source(
    application: QApplication, frame: Frame, install_captures: InstallCaptures
) -> None:
    camera, clip = running(frame), running(frame)
    window = window_over([camera, clip], install_captures)
    window.start()

    choose_file(window, str(CLIP))
    window._show_next_frame()

    assert len(clip.delivered) == 1, "frames are not coming from the new source"
    assert window._timer.isActive()


def test_closing_after_a_swap_releases_the_source_in_use(
    application: QApplication, frame: Frame, install_captures: InstallCaptures
) -> None:
    camera, clip = running(frame), running(frame)
    window = window_over([camera, clip], install_captures)
    choose_file(window, str(CLIP))

    window.close()

    assert (camera.released, clip.released) == (1, 1), "a handle survived the close"


def test_choosing_the_source_already_running_changes_nothing(
    application: QApplication, frame: Frame, install_captures: InstallCaptures
) -> None:
    camera = running(frame)
    window = window_over([camera], install_captures)
    before = window._session.camera

    window._camera_action.trigger()

    assert window._session.camera is before
    assert camera.released == 0


def test_a_clip_starting_over_lets_go_of_held_faces(
    application: QApplication, frame: Frame, install_captures: InstallCaptures
) -> None:
    install_captures([FakeCapture(frames=[frame], opened=True, fps=25.0)])
    window = VideoWindow(Camera(CameraConfig(source=CLIP)).__enter__(), FakePoseSource(()))
    forgotten: list[bool] = []
    window._tracker.reset = lambda: forgotten.append(True)  # type: ignore[method-assign]

    window._show_next_frame()
    assert forgotten == [], "nothing was replayed yet"

    window._show_next_frame()
    assert forgotten == [True], "tracks from the last frame bled into the replay"


def test_frames_that_are_not_replays_keep_the_tracks(
    application: QApplication, frame: Frame, install_captures: InstallCaptures
) -> None:
    install_captures([FakeCapture(frames=[frame] * 6, opened=True, fps=25.0)])
    window = VideoWindow(Camera(CameraConfig(source=CLIP)).__enter__(), FakePoseSource(()))
    forgotten: list[bool] = []
    window._tracker.reset = lambda: forgotten.append(True)  # type: ignore[method-assign]

    for _ in range(4):
        window._show_next_frame()

    assert forgotten == []


# --- choosing how faces are hidden, and whether the skeleton is drawn --------------------


def recording_window(
    frame: Frame, install: InstallCaptures, monkeypatch: pytest.MonkeyPatch
) -> tuple[VideoWindow, list[HideStyle], list[int]]:
    """A window whose pipeline records the style used and the skeletons drawn."""
    install([FakeCapture(frames=[frame] * 12, opened=True, fps=25.0)])
    styles: list[HideStyle] = []
    skeletons: list[int] = []

    def record_hide(target: Frame, regions: Sequence[object], style: HideStyle) -> Frame:
        styles.append(style)
        return target

    def record_draw(target: Frame, people: Sequence[object]) -> Frame:
        skeletons.append(len(people))
        return target

    monkeypatch.setattr("evidone_pose_detection.ui.window.hide_faces", record_hide)
    monkeypatch.setattr("evidone_pose_detection.ui.window.draw_poses", record_draw)
    window = VideoWindow(
        Camera(CameraConfig(source=0)).__enter__(), FakePoseSource((make_person(0.8),))
    )
    return window, styles, skeletons


def test_every_style_has_a_label_for_the_menu() -> None:
    assert set(CONFIG.labels.styles) == set(HideStyle)


def test_the_menu_offers_every_style(
    application: QApplication, frame: Frame, install_captures: InstallCaptures
) -> None:
    window = window_over([running(frame)], install_captures)

    assert window._hiding_menu.title() == CONFIG.labels.hiding
    labels = [action.text() for action in window._hiding_menu.actions()]
    assert labels == [CONFIG.labels.styles[style] for style in HideStyle]


def test_the_app_starts_on_blur_and_says_so(
    application: QApplication, frame: Frame, install_captures: InstallCaptures
) -> None:
    window = window_over([running(frame)], install_captures)

    assert window._settings.hide_style is HideStyle.BLUR
    assert window._style_actions[HideStyle.BLUR].isChecked()
    assert not window._style_actions[HideStyle.PIXELATE].isChecked()
    assert not window._style_actions[HideStyle.FILL].isChecked()


@pytest.mark.parametrize("style", list(HideStyle), ids=str)
def test_choosing_a_style_is_recorded_and_marked(
    application: QApplication, frame: Frame, install_captures: InstallCaptures, style: HideStyle
) -> None:
    window = window_over([running(frame)], install_captures)

    window._style_actions[style].trigger()

    assert window._settings.hide_style is style
    checked = [name for name, action in window._style_actions.items() if action.isChecked()]
    assert checked == [style], "the choice should be exclusive"


@pytest.mark.parametrize("style", list(HideStyle), ids=str)
def test_the_chosen_style_reaches_the_very_next_frame(
    application: QApplication,
    frame: Frame,
    install_captures: InstallCaptures,
    monkeypatch: pytest.MonkeyPatch,
    style: HideStyle,
) -> None:
    window, styles, _ = recording_window(frame, install_captures, monkeypatch)
    window._show_next_frame()

    window._style_actions[style].trigger()
    window._show_next_frame()

    assert styles[0] is HideStyle.BLUR
    assert styles[1] is style


def test_changing_the_style_does_not_disturb_the_running_source(
    application: QApplication, frame: Frame, install_captures: InstallCaptures
) -> None:
    capture = running(frame)
    window = window_over([capture], install_captures)
    camera = window._session.camera
    tracker = window._tracker

    window._style_actions[HideStyle.FILL].trigger()

    assert window._session.camera is camera, "the source was rebuilt"
    assert window._tracker is tracker, "the tracker was rebuilt"
    assert capture.released == 0


def test_the_skeleton_starts_switched_on(
    application: QApplication, frame: Frame, install_captures: InstallCaptures
) -> None:
    window = window_over([running(frame)], install_captures)

    assert window._settings.draw_skeleton is True
    assert window._skeleton_action.isChecked()
    assert window._skeleton_action.text() == CONFIG.labels.skeleton


def test_switching_the_skeleton_off_stops_it_being_drawn(
    application: QApplication,
    frame: Frame,
    install_captures: InstallCaptures,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, _, skeletons = recording_window(frame, install_captures, monkeypatch)
    window._show_next_frame()

    window._skeleton_action.setChecked(False)
    window._show_next_frame()

    assert len(skeletons) == 1, "the skeleton was drawn after being switched off"
    assert window._settings.draw_skeleton is False


def test_switching_the_skeleton_back_on_draws_it_again(
    application: QApplication,
    frame: Frame,
    install_captures: InstallCaptures,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, _, skeletons = recording_window(frame, install_captures, monkeypatch)
    window._skeleton_action.setChecked(False)
    window._show_next_frame()

    window._skeleton_action.setChecked(True)
    window._show_next_frame()

    assert skeletons == [1]
    assert window._skeleton_action.isChecked()


def test_hiding_keeps_running_while_the_skeleton_is_off(
    application: QApplication,
    frame: Frame,
    install_captures: InstallCaptures,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window, styles, skeletons = recording_window(frame, install_captures, monkeypatch)

    window._skeleton_action.setChecked(False)
    window._show_next_frame()

    assert skeletons == []
    assert styles == [HideStyle.BLUR], "faces stopped being hidden with the overlay"


# --- recording the processed video ------------------------------------------------------


@pytest.fixture
def recordings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep every recording the tests make out of the working directory."""
    monkeypatch.setattr(
        "evidone_pose_detection.ui.window._recording_path", lambda: tmp_path / "clip.mp4"
    )
    return tmp_path


def test_recording_is_off_until_it_is_switched_on(
    application: QApplication, frame: Frame, install_captures: InstallCaptures
) -> None:
    window = window_over([running(frame)], install_captures)

    assert window._recorder.is_recording is False
    assert not window._record_action.isChecked()
    assert window._record_action.text() == CONFIG.labels.record


def test_switching_recording_on_opens_a_recording(
    application: QApplication, frame: Frame, install_captures: InstallCaptures
) -> None:
    window = window_over([running(frame)], install_captures)

    window._record_action.setChecked(True)

    assert window._recorder.is_recording is True
    assert window._recorder.destination is not None
    assert window._recorder.destination.suffix == ".mp4"
    assert window._record_action.isChecked()


def test_switching_recording_off_closes_it(
    application: QApplication, frame: Frame, install_captures: InstallCaptures
) -> None:
    window = window_over([running(frame)], install_captures)
    window._record_action.setChecked(True)

    window._record_action.setChecked(False)

    assert window._recorder.is_recording is False
    assert not window._record_action.isChecked()


def test_nothing_is_written_while_recording_is_off(
    application: QApplication, frame: Frame, install_captures: InstallCaptures
) -> None:
    window = window_over([running(frame)], install_captures)
    written: list[Frame] = []

    def capture_frame(frame: Frame) -> None:
        written.append(frame)

    window._recorder.write = capture_frame  # type: ignore[method-assign]

    for _ in range(3):
        window._show_next_frame()

    assert [f for f in written if window._recorder.is_recording] == []


def test_the_recorded_frame_is_the_one_the_window_shows(
    application: QApplication, frame: Frame, install_captures: InstallCaptures
) -> None:
    install_captures([FakeCapture(frames=[frame] * 4, opened=True, fps=25.0)])
    window = VideoWindow(
        Camera(CameraConfig(source=0)).__enter__(), FakePoseSource((make_person(0.8),))
    )
    written: list[Frame] = []

    def capture_frame(frame: Frame) -> None:
        written.append(frame.copy())

    window._recorder.write = capture_frame  # type: ignore[method-assign]

    window._show_next_frame()

    assert len(written) == 1
    assert (written[0] != 128).any(), "the raw capture was recorded, not the processed frame"


def test_the_recording_is_written_after_everything_is_drawn(
    application: QApplication,
    frame: Frame,
    install_captures: InstallCaptures,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_captures([FakeCapture(frames=[frame] * 4, opened=True, fps=25.0)])
    order: list[str] = []

    def record_hide(target: Frame, regions: Sequence[object], style: object) -> Frame:
        order.append("hide")
        return target

    def record_draw(target: Frame, people: Sequence[object]) -> Frame:
        order.append("draw")
        return target

    def record_stats(target: Frame, stats: object) -> Frame:
        order.append("stats")
        return target

    monkeypatch.setattr("evidone_pose_detection.ui.window.hide_faces", record_hide)
    monkeypatch.setattr("evidone_pose_detection.ui.window.draw_poses", record_draw)
    monkeypatch.setattr("evidone_pose_detection.ui.window.draw_stats", record_stats)
    window = VideoWindow(
        Camera(CameraConfig(source=0)).__enter__(), FakePoseSource((make_person(0.8),))
    )

    def note_recording(frame: Frame) -> None:
        order.append("record")

    window._recorder.write = note_recording  # type: ignore[method-assign]

    window._show_next_frame()

    assert order == ["hide", "draw", "stats", "record"]


def test_closing_the_window_closes_the_recording(
    application: QApplication, frame: Frame, install_captures: InstallCaptures, recordings: Path
) -> None:
    window = window_over([running(frame)], install_captures)
    window._record_action.setChecked(True)
    window._show_next_frame()

    window.close()

    assert window._recorder.is_recording is False
    assert (recordings / "clip.mp4").exists(), "nothing was written before closing"


def test_shutdown_closes_the_recording_without_the_window_being_closed(
    application: QApplication, frame: Frame, install_captures: InstallCaptures, recordings: Path
) -> None:
    window = window_over([running(frame)], install_captures)
    window._record_action.setChecked(True)
    window._show_next_frame()

    window.shutdown()

    assert window._recorder.is_recording is False
    assert not window._timer.isActive()


def test_a_recording_that_cannot_be_written_stops_and_is_reported(
    application: QApplication, frame: Frame, install_captures: InstallCaptures
) -> None:
    window = window_over([running(frame)], install_captures)
    reported: list[str] = []
    window._report = lambda title, error: reported.append(title)  # type: ignore[method-assign]
    window._record_action.setChecked(True)
    window._recorder.start(Path("no-such-folder") / "clip.mp4", 25.0)

    window._show_next_frame()

    assert reported == [CONFIG.labels.recording_error_title]
    assert window._recorder.is_recording is False
    assert not window._record_action.isChecked()
    assert window._timer.isActive() or not window._running, "the frame loop died"


# --- logging what the model saw ---------------------------------------------------------


@pytest.fixture
def logs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(
        "evidone_pose_detection.ui.window._detection_log_path", lambda: tmp_path / "run.jsonl"
    )
    return tmp_path / "run.jsonl"


def test_logging_is_off_until_it_is_switched_on(
    application: QApplication, frame: Frame, install_captures: InstallCaptures
) -> None:
    window = window_over([running(frame)], install_captures)

    assert window._detection_log.is_logging is False
    assert not window._log_action.isChecked()
    assert window._log_action.text() == CONFIG.labels.detection_log


def test_switching_logging_on_opens_a_jsonl_file(
    application: QApplication, frame: Frame, install_captures: InstallCaptures, logs: Path
) -> None:
    window = window_over([running(frame)], install_captures)

    window._log_action.setChecked(True)

    assert window._detection_log.is_logging is True
    assert window._detection_log.destination == logs
    assert logs.exists()


def test_switching_logging_off_closes_the_file(
    application: QApplication, frame: Frame, install_captures: InstallCaptures, logs: Path
) -> None:
    window = window_over([running(frame)], install_captures)
    window._log_action.setChecked(True)
    window._show_next_frame()

    window._log_action.setChecked(False)

    assert window._detection_log.is_logging is False
    assert not window._log_action.isChecked()
    assert len(logs.read_text(encoding="utf-8").splitlines()) == 1


def test_nothing_is_logged_while_logging_is_off(
    application: QApplication, frame: Frame, install_captures: InstallCaptures, logs: Path
) -> None:
    window = window_over([running(frame)], install_captures)

    for _ in range(3):
        window._show_next_frame()

    assert not logs.exists()


def test_one_line_is_written_per_shown_frame(
    application: QApplication, frame: Frame, install_captures: InstallCaptures, logs: Path
) -> None:
    install_captures([FakeCapture(frames=[frame] * 6, opened=True, fps=25.0)])
    window = VideoWindow(
        Camera(CameraConfig(source=0)).__enter__(), FakePoseSource((make_person(0.8),))
    )
    window._log_action.setChecked(True)

    for _ in range(3):
        window._show_next_frame()
    window.shutdown()

    records = [json.loads(line) for line in logs.read_text(encoding="utf-8").splitlines()]
    assert [record["frame"] for record in records] == [0, 1, 2]
    assert all(len(record["people"]) == 1 for record in records)


def test_the_log_holds_what_the_model_produced(
    application: QApplication, frame: Frame, install_captures: InstallCaptures, logs: Path
) -> None:
    install_captures([FakeCapture(frames=[frame] * 4, opened=True, fps=25.0)])
    window = VideoWindow(
        Camera(CameraConfig(source=0)).__enter__(), FakePoseSource((make_person(0.8),))
    )
    window._log_action.setChecked(True)

    window._show_next_frame()
    window.shutdown()

    (record,) = [json.loads(line) for line in logs.read_text(encoding="utf-8").splitlines()]
    (logged,) = record["people"]
    assert logged["score"] == pytest.approx(0.8)
    assert len(logged["keypoints"]) == 17


def test_closing_the_window_closes_the_log(
    application: QApplication, frame: Frame, install_captures: InstallCaptures, logs: Path
) -> None:
    window = window_over([running(frame)], install_captures)
    window._log_action.setChecked(True)
    window._show_next_frame()

    window.close()

    assert window._detection_log.is_logging is False


def test_shutdown_closes_the_log(
    application: QApplication, frame: Frame, install_captures: InstallCaptures, logs: Path
) -> None:
    window = window_over([running(frame)], install_captures)
    window._log_action.setChecked(True)
    window._show_next_frame()

    window.shutdown()

    assert window._detection_log.is_logging is False


def test_a_log_that_cannot_be_opened_is_reported_and_stays_off(
    application: QApplication,
    frame: Frame,
    install_captures: InstallCaptures,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "evidone_pose_detection.ui.window._detection_log_path",
        lambda: tmp_path / "no-such-folder" / "run.jsonl",
    )
    window = window_over([running(frame)], install_captures)
    reported: list[str] = []
    window._report = lambda title, error: reported.append(title)  # type: ignore[method-assign]

    window._log_action.setChecked(True)

    assert reported == [CONFIG.labels.log_error_title]
    assert window._detection_log.is_logging is False
    assert not window._log_action.isChecked()


def test_recording_and_logging_count_frames_together(
    application: QApplication,
    frame: Frame,
    install_captures: InstallCaptures,
    logs: Path,
    recordings: Path,
) -> None:
    install_captures([FakeCapture(frames=[frame] * 8, opened=True, fps=25.0)])
    window = VideoWindow(
        Camera(CameraConfig(source=0)).__enter__(), FakePoseSource((make_person(0.8),))
    )
    recorded: list[int] = []
    original_write = window._recorder.write

    def count_frame(frame: Frame) -> None:
        recorded.append(1)
        original_write(frame)

    window._recorder.write = count_frame  # type: ignore[method-assign]
    window._record_action.setChecked(True)
    window._log_action.setChecked(True)

    for _ in range(4):
        window._show_next_frame()
    window.shutdown()

    logged = [json.loads(line) for line in logs.read_text(encoding="utf-8").splitlines()]
    assert len(logged) == len(recorded) == 4, "the log and the recording do not line up"
    assert [record["frame"] for record in logged] == [0, 1, 2, 3]
    assert (recordings / "clip.mp4").stat().st_size > 0
