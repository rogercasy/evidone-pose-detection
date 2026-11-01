from collections.abc import Callable, Sequence

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from evidone_pose_detection.models import (
    KEYPOINT_COUNT,
    BoundingBox,
    Frame,
    InferenceTiming,
    Keypoint,
    PersonPose,
    PoseResult,
)

FRAME_WIDTH = 64
FRAME_HEIGHT = 48
FRAME_FPS = 25.0


class FakeCapture:
    def __init__(self, frames: Sequence[Frame | None], opened: bool, fps: float) -> None:
        self._frames = list(frames)
        self._opened = opened
        self._fps = fps
        self.released = 0
        self.delivered: list[Frame] = []
        self.rewinds = 0
        self._original = list(frames)

    def isOpened(self) -> bool:
        return self._opened

    def get(self, prop: int) -> float:
        return {
            cv2.CAP_PROP_FPS: self._fps,
            cv2.CAP_PROP_FRAME_WIDTH: float(FRAME_WIDTH),
            cv2.CAP_PROP_FRAME_HEIGHT: float(FRAME_HEIGHT),
        }[prop]

    def read(self) -> tuple[bool, Frame | None]:
        if not self._frames:
            return False, None
        captured = self._frames.pop(0)
        if captured is None:
            return False, None
        # cv2.VideoCapture allocates a fresh buffer per read; never hand back the same one.
        frame = captured.copy()
        self.delivered.append(frame)
        return True, frame

    def set(self, prop: int, value: float) -> bool:
        if prop == cv2.CAP_PROP_POS_FRAMES and value == 0:
            self.rewinds += 1
            self._frames = list(self._original)
        return True

    def release(self) -> None:
        self.released += 1


@pytest.fixture
def frame() -> Frame:
    return np.full((FRAME_HEIGHT, FRAME_WIDTH, 3), 128, dtype=np.uint8)


@pytest.fixture
def working_capture(frame: Frame) -> FakeCapture:
    return FakeCapture(frames=[frame] * 4, opened=True, fps=FRAME_FPS)


@pytest.fixture
def install_capture(monkeypatch: pytest.MonkeyPatch) -> Callable[[FakeCapture], list[int]]:
    def install(capture: FakeCapture) -> list[int]:
        requested: list[int] = []

        def video_capture(index: int) -> FakeCapture:
            requested.append(index)
            return capture

        monkeypatch.setattr("evidone_pose_detection.camera.cv2.VideoCapture", video_capture)
        return requested

    return install


@pytest.fixture(scope="session")
def application() -> QApplication:
    existing = QApplication.instance()
    if isinstance(existing, QApplication):
        return existing
    return QApplication([])


def make_person(confidence: float) -> PersonPose:
    return PersonPose(
        box=BoundingBox(left=10.0, top=20.0, right=110.0, bottom=220.0),
        confidence=confidence,
        keypoints=tuple(
            Keypoint(x=float(index), y=float(index * 2), confidence=0.9)
            for index in range(KEYPOINT_COUNT)
        ),
    )


FAKE_TIMING = InferenceTiming(preprocess_ms=0.5, inference_ms=12.5, postprocess_ms=0.25)


class FakePoseSource:
    def __init__(self, people: tuple[PersonPose, ...]) -> None:
        self._people = people
        self.seen_frames = 0
        self.last_frame: Frame | None = None
        self.timing = FAKE_TIMING

    def estimate(self, frame: Frame) -> PoseResult:
        self.seen_frames += 1
        self.last_frame = frame.copy()
        return PoseResult(people=self._people, timing=self.timing)


@pytest.fixture
def no_poses() -> FakePoseSource:
    return FakePoseSource(())
