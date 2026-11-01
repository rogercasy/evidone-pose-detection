from collections.abc import Callable, Sequence

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from conftest import FRAME_HEIGHT, FRAME_WIDTH, FakeCapture, FakePoseSource, make_person
from evidone_pose_detection.camera import Camera
from evidone_pose_detection.models import CameraConfig, Frame, PersonPose
from evidone_pose_detection.ui.window import MINIMUM_YIELD_MS, WINDOW_TITLE, VideoWindow
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
    return VideoWindow(Camera(CameraConfig(device_index=0)).__enter__(), no_poses)


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
    window = VideoWindow(Camera(CameraConfig(device_index=0)).__enter__(), no_poses)

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
    window = VideoWindow(Camera(CameraConfig(device_index=0)).__enter__(), poses)

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
    clock = iter([0.0, 10.0])
    monkeypatch.setattr("evidone_pose_detection.ui.window.time.perf_counter", lambda: next(clock))

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
    window = VideoWindow(Camera(CameraConfig(device_index=0)).__enter__(), FakePoseSource(()))
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
    window = VideoWindow(Camera(CameraConfig(device_index=0)).__enter__(), poses)

    window._show_next_frame()

    assert (working_capture.delivered[-1] != 128).any()


def test_the_model_never_sees_a_frame_we_have_drawn_on(
    application: QApplication, working_capture: FakeCapture, install_capture: InstallCapture
) -> None:
    install_capture(working_capture)
    poses = FakePoseSource((make_person(0.8),))
    window = VideoWindow(Camera(CameraConfig(device_index=0)).__enter__(), poses)

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
        def estimate(self, frame: Frame) -> tuple[PersonPose, ...]:
            order.append("estimate")
            return super().estimate(frame)

    def record_blur(frame: Frame, people: Sequence[PersonPose]) -> Frame:
        order.append("blur")
        return frame

    def record_draw(frame: Frame, people: Sequence[PersonPose]) -> Frame:
        order.append("draw")
        return frame

    monkeypatch.setattr("evidone_pose_detection.ui.window.blur_faces", record_blur)
    monkeypatch.setattr("evidone_pose_detection.ui.window.draw_poses", record_draw)
    window = VideoWindow(
        Camera(CameraConfig(device_index=0)).__enter__(), RecordingPoses((make_person(0.8),))
    )

    window._show_next_frame()

    assert order == ["estimate", "blur", "draw"]


def test_the_skeleton_stays_sharp_on_top_of_a_blurred_face(
    application: QApplication, working_capture: FakeCapture, install_capture: InstallCapture
) -> None:
    install_capture(working_capture)
    poses = FakePoseSource((make_person(0.8),))
    window = VideoWindow(Camera(CameraConfig(device_index=0)).__enter__(), poses)

    window._show_next_frame()

    shown = working_capture.delivered[-1]
    drawn_colours = {tuple(int(c) for c in colour) for colour in REGION_COLOURS.values()}
    painted = {tuple(int(c) for c in pixel) for row in shown for pixel in row}
    assert drawn_colours & painted, "no un-smeared skeleton colour survived the blur"
