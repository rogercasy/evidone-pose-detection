from collections.abc import Callable

import pytest

from conftest import FRAME_FPS, FRAME_HEIGHT, FRAME_WIDTH, FakeCapture
from evidone_pose_detection.camera import FALLBACK_FPS, Camera
from evidone_pose_detection.errors import CameraNotOpenError, CameraOpenError
from evidone_pose_detection.models import CameraConfig, Frame

InstallCapture = Callable[[FakeCapture], list[int]]


@pytest.mark.parametrize("device_index", [0, 2])
def test_open_uses_the_configured_device_index(
    working_capture: FakeCapture, install_capture: InstallCapture, device_index: int
) -> None:
    requested = install_capture(working_capture)

    with Camera(CameraConfig(device_index=device_index)):
        assert requested == [device_index]


def test_open_raises_when_the_device_is_unavailable(install_capture: InstallCapture) -> None:
    capture = FakeCapture(frames=[], opened=False, fps=FRAME_FPS)
    install_capture(capture)

    with pytest.raises(CameraOpenError) as error:
        Camera(CameraConfig(device_index=7)).open()

    assert error.value.device_index == 7


def test_open_releases_the_device_it_could_not_use(install_capture: InstallCapture) -> None:
    capture = FakeCapture(frames=[], opened=False, fps=FRAME_FPS)
    install_capture(capture)

    with pytest.raises(CameraOpenError):
        Camera(CameraConfig(device_index=0)).open()

    assert capture.released == 1


def test_properties_report_the_capture_geometry_and_rate(
    working_capture: FakeCapture, install_capture: InstallCapture
) -> None:
    install_capture(working_capture)

    with Camera(CameraConfig(device_index=0)) as camera:
        properties = camera.properties()

    assert (properties.width, properties.height, properties.fps) == (
        FRAME_WIDTH,
        FRAME_HEIGHT,
        FRAME_FPS,
    )


@pytest.mark.parametrize("reported_fps", [0.0, -1.0])
def test_properties_fall_back_when_the_device_reports_no_rate(
    install_capture: InstallCapture, reported_fps: float
) -> None:
    install_capture(FakeCapture(frames=[], opened=True, fps=reported_fps))

    with Camera(CameraConfig(device_index=0)) as camera:
        assert camera.properties().fps == FALLBACK_FPS


def test_read_returns_the_captured_frame(
    frame: Frame, working_capture: FakeCapture, install_capture: InstallCapture
) -> None:
    install_capture(working_capture)

    with Camera(CameraConfig(device_index=0)) as camera:
        captured = camera.read()

    assert captured is not None
    assert (captured == frame).all()


def test_read_returns_none_when_the_device_stops_delivering(
    install_capture: InstallCapture,
) -> None:
    install_capture(FakeCapture(frames=[], opened=True, fps=FRAME_FPS))

    with Camera(CameraConfig(device_index=0)) as camera:
        assert camera.read() is None


@pytest.mark.parametrize("use", [Camera.read, Camera.properties])
def test_using_a_closed_camera_raises(use: Callable[[Camera], object]) -> None:
    with pytest.raises(CameraNotOpenError):
        use(Camera(CameraConfig(device_index=0)))


def test_leaving_the_context_releases_the_device(
    working_capture: FakeCapture, install_capture: InstallCapture
) -> None:
    install_capture(working_capture)

    with Camera(CameraConfig(device_index=0)):
        assert working_capture.released == 0

    assert working_capture.released == 1


def test_release_is_idempotent(
    working_capture: FakeCapture, install_capture: InstallCapture
) -> None:
    install_capture(working_capture)
    camera = Camera(CameraConfig(device_index=0))
    camera.open()

    camera.release()
    camera.release()

    assert working_capture.released == 1


def test_a_capture_backend_that_explodes_becomes_our_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def explode(index: int) -> FakeCapture:
        message = "OpenCV(5.0.0) failed to query the v4l2 backend"
        raise RuntimeError(message)

    monkeypatch.setattr("evidone_pose_detection.camera.cv2.VideoCapture", explode)

    with pytest.raises(CameraOpenError) as error:
        Camera(CameraConfig(device_index=3)).open()

    assert error.value.device_index == 3
    assert isinstance(error.value.__cause__, RuntimeError)


def test_the_camera_error_carries_a_message_of_its_own() -> None:
    message = str(CameraOpenError(2))

    assert "2" in message
    assert message != ""
    assert "Traceback" not in message
