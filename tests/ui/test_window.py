from collections.abc import Callable, Sequence
from pathlib import Path

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication, QToolBar, QToolButton

from conftest import FRAME_HEIGHT, FRAME_WIDTH, FakeCapture, FakePoseSource, make_person
from evidone_pose_detection.camera import Camera
from evidone_pose_detection.models import CameraConfig, Frame, HideStyle, PersonPose, PoseResult
from evidone_pose_detection.ui.window import (
    CAMERA_LABEL,
    HIDING_LABEL,
    MINIMUM_YIELD_MS,
    SETTINGS_LABEL,
    SKELETON_LABEL,
    SOURCE_LABEL,
    STYLE_LABELS,
    VIDEO_LABEL,
    WINDOW_TITLE,
    VideoWindow,
)
from evidone_pose_detection.vision.rendering import REGION_COLOURS

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
    assert window.windowTitle().startswith(WINDOW_TITLE)
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

    assert window.windowTitle() == f"{WINDOW_TITLE} \u2014 {people} detected"


def test_start_schedules_a_single_frame_at_the_capture_rate(window: VideoWindow) -> None:
    window.start()

    assert window._timer.isActive()
    assert window._timer.isSingleShot()
    assert window._timer.interval() == 40


def test_each_frame_rearms_the_next_one(window: VideoWindow) -> None:
    window.start()

    window._show_next_frame()

    assert window._timer.isActive()
    assert MINIMUM_YIELD_MS <= window._timer.interval() <= 40


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

    assert window._timer.interval() == MINIMUM_YIELD_MS


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
    drawn_colours = {tuple(int(c) for c in colour) for colour in REGION_COLOURS.values()}
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
    window._report_source_error = lambda error: reported.append(str(error))  # type: ignore[method-assign]
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
    assert any(button.text() == SETTINGS_LABEL for button in buttons)


def test_the_settings_menu_offers_a_camera_and_a_video_file(
    application: QApplication, frame: Frame, install_captures: InstallCaptures
) -> None:
    window = window_over([running(frame)], install_captures)

    assert window._source_menu.title() == SOURCE_LABEL
    labels = [action.text() for action in window._source_menu.actions()]
    assert labels[0] == CAMERA_LABEL
    assert labels[1].startswith(VIDEO_LABEL)


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
    assert window._video_action.text() == VIDEO_LABEL


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
    assert set(STYLE_LABELS) == set(HideStyle)


def test_the_menu_offers_every_style(
    application: QApplication, frame: Frame, install_captures: InstallCaptures
) -> None:
    window = window_over([running(frame)], install_captures)

    assert window._hiding_menu.title() == HIDING_LABEL
    labels = [action.text() for action in window._hiding_menu.actions()]
    assert labels == [STYLE_LABELS[style] for style in HideStyle]


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
    assert window._skeleton_action.text() == SKELETON_LABEL


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
