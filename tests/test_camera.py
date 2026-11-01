from collections.abc import Callable, Iterator
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
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

    with Camera(CameraConfig(source=device_index)):
        assert requested == [device_index]


def test_open_raises_when_the_device_is_unavailable(install_capture: InstallCapture) -> None:
    capture = FakeCapture(frames=[], opened=False, fps=FRAME_FPS)
    install_capture(capture)

    with pytest.raises(CameraOpenError) as error:
        Camera(CameraConfig(source=7)).open()

    assert error.value.source == 7


def test_open_releases_the_device_it_could_not_use(install_capture: InstallCapture) -> None:
    capture = FakeCapture(frames=[], opened=False, fps=FRAME_FPS)
    install_capture(capture)

    with pytest.raises(CameraOpenError):
        Camera(CameraConfig(source=0)).open()

    assert capture.released == 1


def test_properties_report_the_capture_geometry_and_rate(
    working_capture: FakeCapture, install_capture: InstallCapture
) -> None:
    install_capture(working_capture)

    with Camera(CameraConfig(source=0)) as camera:
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

    with Camera(CameraConfig(source=0)) as camera:
        assert camera.properties().fps == FALLBACK_FPS


def test_read_returns_the_captured_frame(
    frame: Frame, working_capture: FakeCapture, install_capture: InstallCapture
) -> None:
    install_capture(working_capture)

    with Camera(CameraConfig(source=0)) as camera:
        captured = camera.read()

    assert captured is not None
    assert (captured == frame).all()


def test_read_returns_none_when_the_device_stops_delivering(
    install_capture: InstallCapture,
) -> None:
    install_capture(FakeCapture(frames=[], opened=True, fps=FRAME_FPS))

    with Camera(CameraConfig(source=0)) as camera:
        assert camera.read() is None


@pytest.mark.parametrize("use", [Camera.read, Camera.properties])
def test_using_a_closed_camera_raises(use: Callable[[Camera], object]) -> None:
    with pytest.raises(CameraNotOpenError):
        use(Camera(CameraConfig(source=0)))


def test_leaving_the_context_releases_the_device(
    working_capture: FakeCapture, install_capture: InstallCapture
) -> None:
    install_capture(working_capture)

    with Camera(CameraConfig(source=0)):
        assert working_capture.released == 0

    assert working_capture.released == 1


def test_release_is_idempotent(
    working_capture: FakeCapture, install_capture: InstallCapture
) -> None:
    install_capture(working_capture)
    camera = Camera(CameraConfig(source=0))
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
        Camera(CameraConfig(source=3)).open()

    assert error.value.source == 3
    assert isinstance(error.value.__cause__, RuntimeError)


def test_the_camera_error_carries_a_message_of_its_own() -> None:
    message = str(CameraOpenError(2))

    assert "2" in message
    assert message != ""
    assert "Traceback" not in message


CLIP_FRAMES = 4
CLIP_WIDTH = 64
CLIP_HEIGHT = 48


def write_clip(destination: Path, frames: int) -> Path:
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter.fourcc(*"MJPG"), 10.0, (CLIP_WIDTH, CLIP_HEIGHT)
    )
    assert writer.isOpened(), "the test could not create a clip to read back"
    for index in range(frames):
        writer.write(np.full((CLIP_HEIGHT, CLIP_WIDTH, 3), index * 50, dtype=np.uint8))
    writer.release()
    return destination


@pytest.fixture
def clip() -> Iterator[Path]:
    with TemporaryDirectory() as directory:
        yield write_clip(Path(directory) / "clip.avi", frames=CLIP_FRAMES)


def test_a_video_file_opens_through_the_same_wrapper(clip: Path) -> None:
    with Camera(CameraConfig(source=clip)) as camera:
        frame = camera.read()

    assert frame is not None
    assert frame.shape == (CLIP_HEIGHT, CLIP_WIDTH, 3)


def test_a_clip_reports_its_own_geometry(clip: Path) -> None:
    with Camera(CameraConfig(source=clip)) as camera:
        properties = camera.properties()

    assert (properties.width, properties.height) == (CLIP_WIDTH, CLIP_HEIGHT)
    assert properties.fps > 0


def test_a_clip_keeps_playing_past_its_last_frame(clip: Path) -> None:
    with Camera(CameraConfig(source=clip)) as camera:
        frames = [camera.read() for _ in range(CLIP_FRAMES * 3)]

    assert all(frame is not None for frame in frames), "the clip stopped at its end"


def test_a_clip_starts_again_from_the_beginning(clip: Path) -> None:
    with Camera(CameraConfig(source=clip)) as camera:
        first = camera.read()
        for _ in range(CLIP_FRAMES - 1):
            camera.read()
        after_the_end = camera.read()

    assert first is not None
    assert after_the_end is not None
    assert int(after_the_end.mean()) == pytest.approx(int(first.mean()), abs=2)


def test_a_missing_file_raises_the_same_error_as_a_missing_camera() -> None:
    missing = Path("/nonexistent/no-such-clip.avi")

    with pytest.raises(CameraOpenError) as error:
        Camera(CameraConfig(source=missing)).open()

    assert error.value.source == missing
    assert str(missing) in str(error.value)


def test_a_file_that_is_not_a_video_names_itself_in_the_error() -> None:
    with TemporaryDirectory() as directory:
        rubbish = Path(directory) / "not-a-video.avi"
        rubbish.write_bytes(b"this is not a video")

        with pytest.raises(CameraOpenError) as error:
            Camera(CameraConfig(source=rubbish)).open()

        assert "not-a-video.avi" in str(error.value)


def test_the_error_type_is_shared_by_both_kinds_of_source() -> None:
    device = CameraOpenError(3)
    clip = CameraOpenError(Path("clips/sample.avi"))

    assert type(device) is type(clip)
    assert "camera device 3" in str(device)
    assert "clips/sample.avi" in str(clip)


def test_a_clip_rewinds_when_it_runs_out(install_capture: InstallCapture, frame: Frame) -> None:
    capture = FakeCapture(frames=[frame], opened=True, fps=25.0)
    install_capture(capture)

    with Camera(CameraConfig(source=Path("clip.avi"))) as camera:
        camera.read()
        replayed = camera.read()

    assert capture.rewinds == 1
    assert replayed is not None


def test_a_camera_that_stops_delivering_is_never_rewound(
    install_capture: InstallCapture, frame: Frame
) -> None:
    capture = FakeCapture(frames=[frame], opened=True, fps=25.0)
    install_capture(capture)

    with Camera(CameraConfig(source=0)) as camera:
        camera.read()
        stopped = camera.read()

    assert capture.rewinds == 0, "a live camera must not be rewound"
    assert stopped is None


def test_a_clip_with_nothing_to_replay_gives_up(install_capture: InstallCapture) -> None:
    capture = FakeCapture(frames=[], opened=True, fps=25.0)
    install_capture(capture)

    with Camera(CameraConfig(source=Path("empty.avi"))) as camera:
        assert camera.read() is None

    assert capture.rewinds == 1, "it should try once before giving up"
