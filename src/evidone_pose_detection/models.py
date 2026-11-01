from dataclasses import dataclass
from enum import IntEnum
from typing import Protocol

import cv2

type Frame = cv2.typing.MatLike


class COCOKeypoint(IntEnum):
    NOSE = 0
    LEFT_EYE = 1
    RIGHT_EYE = 2
    LEFT_EAR = 3
    RIGHT_EAR = 4
    LEFT_SHOULDER = 5
    RIGHT_SHOULDER = 6
    LEFT_ELBOW = 7
    RIGHT_ELBOW = 8
    LEFT_WRIST = 9
    RIGHT_WRIST = 10
    LEFT_HIP = 11
    RIGHT_HIP = 12
    LEFT_KNEE = 13
    RIGHT_KNEE = 14
    LEFT_ANKLE = 15
    RIGHT_ANKLE = 16


KEYPOINT_COUNT = len(COCOKeypoint)


@dataclass(frozen=True, slots=True)
class CameraConfig:
    device_index: int


@dataclass(frozen=True, slots=True)
class PoseConfig:
    weights: str
    confidence: float


@dataclass(frozen=True, slots=True)
class AppConfig:
    camera: CameraConfig
    pose: PoseConfig


@dataclass(frozen=True, slots=True)
class CaptureProperties:
    width: int
    height: int
    fps: float

    @property
    def frame_interval_ms(self) -> int:
        return round(1000 / self.fps)


@dataclass(frozen=True, slots=True)
class BoundingBox:
    left: float
    top: float
    right: float
    bottom: float


@dataclass(frozen=True, slots=True)
class Keypoint:
    x: float
    y: float
    confidence: float


@dataclass(frozen=True, slots=True)
class PersonPose:
    box: BoundingBox
    confidence: float
    keypoints: tuple[Keypoint, ...]

    def keypoint(self, name: COCOKeypoint) -> Keypoint:
        return self.keypoints[name]


class PoseSource(Protocol):
    def estimate(self, frame: Frame) -> tuple[PersonPose, ...]: ...
