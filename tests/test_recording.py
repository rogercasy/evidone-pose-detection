from collections.abc import Iterator
from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np
import pytest

from evidone_pose_detection.errors import RecordingError
from evidone_pose_detection.models import Frame
from evidone_pose_detection.recording import Recorder

WIDTH = 64
HEIGHT = 48
FPS = 20.0


@pytest.fixture
def folder() -> Iterator[Path]:
    with TemporaryDirectory() as directory:
        yield Path(directory)


def frame_of(width: int, height: int, value: int) -> Frame:
    return np.full((height, width, 3), value, dtype=np.uint8)


def played_back(destination: Path) -> tuple[int, tuple[int, int]]:
    capture = cv2.VideoCapture(str(destination))
    assert capture.isOpened(), "the recording could not be opened for playback"
    frames = 0
    size = (0, 0)
    while True:
        read, frame = capture.read()
        if not read:
            break
        frames += 1
        size = (frame.shape[1], frame.shape[0])
    capture.release()
    return frames, size


def test_a_recorder_starts_switched_off() -> None:
    recorder = Recorder()

    assert recorder.is_recording is False
    assert recorder.destination is None


def test_writing_while_switched_off_does_nothing(folder: Path) -> None:
    recorder = Recorder()

    recorder.write(frame_of(WIDTH, HEIGHT, 100))

    assert list(folder.iterdir()) == []


def test_the_file_is_only_opened_once_a_frame_has_arrived(folder: Path) -> None:
    destination = folder / "clip.mp4"
    recorder = Recorder()

    recorder.start(destination, FPS)

    assert recorder.is_recording is True
    assert not destination.exists(), "the file was opened before any frame arrived"


def test_the_size_comes_from_the_first_frame(folder: Path) -> None:
    destination = folder / "clip.mp4"
    recorder = Recorder()
    recorder.start(destination, FPS)

    for value in (40, 90, 140):
        recorder.write(frame_of(WIDTH, HEIGHT, value))
    recorder.close()

    frames, size = played_back(destination)
    assert frames == 3
    assert size == (WIDTH, HEIGHT)


def test_a_later_frame_of_another_size_is_fitted_into_the_file(folder: Path) -> None:
    destination = folder / "clip.mp4"
    recorder = Recorder()
    recorder.start(destination, FPS)

    recorder.write(frame_of(WIDTH, HEIGHT, 60))
    recorder.write(frame_of(WIDTH * 3, HEIGHT * 3, 200))
    recorder.write(frame_of(WIDTH // 2, HEIGHT * 2, 120))
    recorder.close()

    frames, size = played_back(destination)
    assert frames == 3
    assert size == (WIDTH, HEIGHT), "the file changed size part way through"


def test_closing_leaves_a_file_that_plays(folder: Path) -> None:
    destination = folder / "clip.mp4"
    recorder = Recorder()
    recorder.start(destination, FPS)
    recorder.write(frame_of(WIDTH, HEIGHT, 70))

    recorder.close()

    assert recorder.is_recording is False
    assert destination.stat().st_size > 0
    assert played_back(destination)[0] == 1


def test_closing_twice_is_harmless(folder: Path) -> None:
    recorder = Recorder()
    recorder.start(folder / "clip.mp4", FPS)
    recorder.write(frame_of(WIDTH, HEIGHT, 70))

    recorder.close()
    recorder.close()

    assert recorder.is_recording is False


def test_closing_without_ever_starting_is_harmless() -> None:
    Recorder().close()


def test_writing_after_closing_does_nothing(folder: Path) -> None:
    destination = folder / "clip.mp4"
    recorder = Recorder()
    recorder.start(destination, FPS)
    recorder.write(frame_of(WIDTH, HEIGHT, 70))
    recorder.close()

    recorder.write(frame_of(WIDTH, HEIGHT, 200))

    assert played_back(destination)[0] == 1


def test_recording_again_writes_a_fresh_file(folder: Path) -> None:
    first, second = folder / "one.mp4", folder / "two.mp4"
    recorder = Recorder()

    recorder.start(first, FPS)
    recorder.write(frame_of(WIDTH, HEIGHT, 70))
    recorder.close()
    recorder.start(second, FPS)
    recorder.write(frame_of(WIDTH * 2, HEIGHT * 2, 150))
    recorder.write(frame_of(WIDTH * 2, HEIGHT * 2, 150))
    recorder.close()

    assert played_back(first) == (1, (WIDTH, HEIGHT))
    assert played_back(second) == (2, (WIDTH * 2, HEIGHT * 2))


def test_starting_again_closes_the_file_already_open(folder: Path) -> None:
    first, second = folder / "one.mp4", folder / "two.mp4"
    recorder = Recorder()
    recorder.start(first, FPS)
    recorder.write(frame_of(WIDTH, HEIGHT, 70))

    recorder.start(second, FPS)
    recorder.write(frame_of(WIDTH, HEIGHT, 70))
    recorder.close()

    assert played_back(first)[0] == 1, "the first file was left unreadable"
    assert played_back(second)[0] == 1


def test_a_destination_that_cannot_be_written_is_reported(folder: Path) -> None:
    destination = folder / "no-such-folder" / "clip.mp4"
    recorder = Recorder()
    recorder.start(destination, FPS)

    with pytest.raises(RecordingError) as error:
        recorder.write(frame_of(WIDTH, HEIGHT, 70))

    assert error.value.destination == destination
    assert str(destination) in str(error.value)
    assert recorder.is_recording is False, "a failed open left recording switched on"
