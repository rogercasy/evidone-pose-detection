from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

import cv2

type Frame = cv2.typing.MatLike

type CaptureSource = int | Path

# The device opened unless another source is asked for.
DEFAULT_CAMERA_INDEX = 0


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

VISIBILITY_THRESHOLD = 0.5


class HideStyle(StrEnum):
    BLUR = "blur"
    PIXELATE = "pixelate"
    FILL = "fill"


DEFAULT_HIDE_STYLE = HideStyle.BLUR


class BodyRegion(StrEnum):
    HEAD = "head"
    TORSO = "torso"
    LIMB = "limb"


@dataclass(frozen=True, slots=True)
class FaceSpan:
    start: COCOKeypoint
    end: COCOKeypoint
    radius_scale: float


@dataclass(frozen=True, slots=True)
class FaceRegion:
    centre_x: float
    centre_y: float
    radius: float


@dataclass(frozen=True, slots=True)
class Bone:
    start: COCOKeypoint
    end: COCOKeypoint
    region: BodyRegion


KEYPOINT_REGIONS: Mapping[COCOKeypoint, BodyRegion] = MappingProxyType(
    {
        COCOKeypoint.NOSE: BodyRegion.HEAD,
        COCOKeypoint.LEFT_EYE: BodyRegion.HEAD,
        COCOKeypoint.RIGHT_EYE: BodyRegion.HEAD,
        COCOKeypoint.LEFT_EAR: BodyRegion.HEAD,
        COCOKeypoint.RIGHT_EAR: BodyRegion.HEAD,
        COCOKeypoint.LEFT_SHOULDER: BodyRegion.TORSO,
        COCOKeypoint.RIGHT_SHOULDER: BodyRegion.TORSO,
        COCOKeypoint.LEFT_HIP: BodyRegion.TORSO,
        COCOKeypoint.RIGHT_HIP: BodyRegion.TORSO,
        COCOKeypoint.LEFT_ELBOW: BodyRegion.LIMB,
        COCOKeypoint.RIGHT_ELBOW: BodyRegion.LIMB,
        COCOKeypoint.LEFT_WRIST: BodyRegion.LIMB,
        COCOKeypoint.RIGHT_WRIST: BodyRegion.LIMB,
        COCOKeypoint.LEFT_KNEE: BodyRegion.LIMB,
        COCOKeypoint.RIGHT_KNEE: BodyRegion.LIMB,
        COCOKeypoint.LEFT_ANKLE: BodyRegion.LIMB,
        COCOKeypoint.RIGHT_ANKLE: BodyRegion.LIMB,
    }
)

SKELETON: tuple[Bone, ...] = (
    Bone(COCOKeypoint.NOSE, COCOKeypoint.LEFT_EYE, BodyRegion.HEAD),
    Bone(COCOKeypoint.NOSE, COCOKeypoint.RIGHT_EYE, BodyRegion.HEAD),
    Bone(COCOKeypoint.LEFT_EYE, COCOKeypoint.RIGHT_EYE, BodyRegion.HEAD),
    Bone(COCOKeypoint.LEFT_EYE, COCOKeypoint.LEFT_EAR, BodyRegion.HEAD),
    Bone(COCOKeypoint.RIGHT_EYE, COCOKeypoint.RIGHT_EAR, BodyRegion.HEAD),
    Bone(COCOKeypoint.LEFT_EAR, COCOKeypoint.LEFT_SHOULDER, BodyRegion.HEAD),
    Bone(COCOKeypoint.RIGHT_EAR, COCOKeypoint.RIGHT_SHOULDER, BodyRegion.HEAD),
    Bone(COCOKeypoint.LEFT_SHOULDER, COCOKeypoint.RIGHT_SHOULDER, BodyRegion.TORSO),
    Bone(COCOKeypoint.LEFT_SHOULDER, COCOKeypoint.LEFT_HIP, BodyRegion.TORSO),
    Bone(COCOKeypoint.RIGHT_SHOULDER, COCOKeypoint.RIGHT_HIP, BodyRegion.TORSO),
    Bone(COCOKeypoint.LEFT_HIP, COCOKeypoint.RIGHT_HIP, BodyRegion.TORSO),
    Bone(COCOKeypoint.LEFT_SHOULDER, COCOKeypoint.LEFT_ELBOW, BodyRegion.LIMB),
    Bone(COCOKeypoint.LEFT_ELBOW, COCOKeypoint.LEFT_WRIST, BodyRegion.LIMB),
    Bone(COCOKeypoint.RIGHT_SHOULDER, COCOKeypoint.RIGHT_ELBOW, BodyRegion.LIMB),
    Bone(COCOKeypoint.RIGHT_ELBOW, COCOKeypoint.RIGHT_WRIST, BodyRegion.LIMB),
    Bone(COCOKeypoint.LEFT_HIP, COCOKeypoint.LEFT_KNEE, BodyRegion.LIMB),
    Bone(COCOKeypoint.LEFT_KNEE, COCOKeypoint.LEFT_ANKLE, BodyRegion.LIMB),
    Bone(COCOKeypoint.RIGHT_HIP, COCOKeypoint.RIGHT_KNEE, BodyRegion.LIMB),
    Bone(COCOKeypoint.RIGHT_KNEE, COCOKeypoint.RIGHT_ANKLE, BodyRegion.LIMB),
)

FACE_LANDMARKS: tuple[COCOKeypoint, ...] = (
    COCOKeypoint.NOSE,
    COCOKeypoint.LEFT_EYE,
    COCOKeypoint.RIGHT_EYE,
    COCOKeypoint.LEFT_EAR,
    COCOKeypoint.RIGHT_EAR,
)

# Each span collapses at a different head angle, so the largest estimate wins.
# The scales are deliberately generous: over-covering only blurs background.
FACE_SPANS: tuple[FaceSpan, ...] = (
    FaceSpan(COCOKeypoint.LEFT_EYE, COCOKeypoint.RIGHT_EYE, 2.0),
    FaceSpan(COCOKeypoint.LEFT_EAR, COCOKeypoint.RIGHT_EAR, 0.9),
    FaceSpan(COCOKeypoint.NOSE, COCOKeypoint.LEFT_EAR, 1.2),
    FaceSpan(COCOKeypoint.NOSE, COCOKeypoint.RIGHT_EAR, 1.2),
    FaceSpan(COCOKeypoint.NOSE, COCOKeypoint.LEFT_EYE, 2.6),
    FaceSpan(COCOKeypoint.NOSE, COCOKeypoint.RIGHT_EYE, 2.6),
    FaceSpan(COCOKeypoint.LEFT_EYE, COCOKeypoint.LEFT_EAR, 1.8),
    FaceSpan(COCOKeypoint.RIGHT_EYE, COCOKeypoint.RIGHT_EAR, 1.8),
)

# Used only when a single face landmark is visible, so no face span can be measured.
SHOULDER_SPAN = FaceSpan(COCOKeypoint.LEFT_SHOULDER, COCOKeypoint.RIGHT_SHOULDER, 0.35)


@dataclass(frozen=True, slots=True)
class CameraConfig:
    source: CaptureSource

    @property
    def is_file(self) -> bool:
        return isinstance(self.source, Path)


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

    @property
    def is_visible(self) -> bool:
        return self.confidence >= VISIBILITY_THRESHOLD


@dataclass(frozen=True, slots=True)
class PersonPose:
    box: BoundingBox
    confidence: float
    keypoints: tuple[Keypoint, ...]

    def keypoint(self, name: COCOKeypoint) -> Keypoint:
        return self.keypoints[name]


@dataclass(frozen=True, slots=True)
class InferenceTiming:
    preprocess_ms: float
    inference_ms: float
    postprocess_ms: float

    @property
    def total_ms(self) -> float:
        return self.preprocess_ms + self.inference_ms + self.postprocess_ms


@dataclass(frozen=True, slots=True)
class PoseResult:
    people: tuple[PersonPose, ...]
    timing: InferenceTiming


@dataclass(frozen=True, slots=True)
class FrameStats:
    inference_ms: float
    frames_per_second: float
    people: int


class PoseSource(Protocol):
    def estimate(self, frame: Frame) -> PoseResult: ...
