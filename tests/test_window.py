from collections.abc import Callable

import pytest
from PySide6.QtCore import QEvent, Qt
from PySide6.QtGui import QKeyEvent
from PySide6.QtWidgets import QApplication

from conftest import FRAME_HEIGHT, FRAME_WIDTH, FakeCapture
from evidone_pose_detection.camera import Camera
from evidone_pose_detection.models import CameraConfig
from evidone_pose_detection.window import WINDOW_TITLE, VideoWindow

InstallCapture = Callable[[FakeCapture], list[int]]


@pytest.fixture
def window(
    application: QApplication, working_capture: FakeCapture, install_capture: InstallCapture
) -> VideoWindow:
    install_capture(working_capture)
    return VideoWindow(Camera(CameraConfig(device_index=0)).__enter__())


def test_window_adopts_the_title_and_the_frame_size(window: VideoWindow) -> None:
    assert window.windowTitle() == WINDOW_TITLE
    assert (window.width(), window.height()) == (FRAME_WIDTH, FRAME_HEIGHT)


def test_start_schedules_frames_at_the_capture_rate(window: VideoWindow) -> None:
    window.start()

    assert window._timer.isActive()
    assert window._timer.interval() == 40


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
    window = VideoWindow(Camera(CameraConfig(device_index=0)).__enter__())
    window.show()
    window.start()

    window._show_next_frame()

    assert not window.isVisible()
    assert capture.released == 1
