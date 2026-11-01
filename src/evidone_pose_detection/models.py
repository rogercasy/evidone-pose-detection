from collections.abc import Mapping
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Protocol

import cv2

type Frame = cv2.typing.MatLike

type CaptureSource = int | Path


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


class HideStyle(StrEnum):
    BLUR = "blur"
    PIXELATE = "pixelate"
    FILL = "fill"


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


type Colour = tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class CameraSettings:
    default_index: int
    fallback_fps: float


@dataclass(frozen=True, slots=True)
class PoseSettings:
    weights: str
    confidence: float


@dataclass(frozen=True, slots=True)
class FaceSettings:
    visibility_threshold: float
    landmark_coverage_margin: float
    shoulder_scale: float
    spans: tuple[FaceSpan, ...]


@dataclass(frozen=True, slots=True)
class TrackingSettings:
    match_distance_ratio: float
    smoothing: float
    hold_frames: int
    hold_growth_per_frame: float


@dataclass(frozen=True, slots=True)
class HidingSettings:
    default_style: HideStyle
    blur_samples: int
    pixel_blocks: int
    fill_colour: Colour


@dataclass(frozen=True, slots=True)
class SkeletonSettings:
    draw_by_default: bool
    joint_radius: int
    bone_thickness: int
    colours: Mapping[BodyRegion, Colour]


@dataclass(frozen=True, slots=True)
class PanelSettings:
    origin: tuple[int, int]
    padding: int
    alpha: float
    colour: Colour
    text_colour: Colour
    font_scale: float
    font_thickness: int
    line_height: int


@dataclass(frozen=True, slots=True)
class WindowSettings:
    title: str
    view_background: str
    minimum_yield_ms: int
    frame_rate_window: int


@dataclass(frozen=True, slots=True)
class OutputSettings:
    stem: str
    fourcc: str


@dataclass(frozen=True, slots=True)
class LabelSettings:
    toolbar: str
    settings: str
    source: str
    camera: str
    video_file: str
    hiding: str
    skeleton: str
    record: str
    detection_log: str
    file_dialog_title: str
    video_file_filter: str
    source_error_title: str
    recording_error_title: str
    log_error_title: str
    styles: Mapping[HideStyle, str]


@dataclass(frozen=True, slots=True)
class Config:
    camera: CameraSettings
    pose: PoseSettings
    faces: FaceSettings
    tracking: TrackingSettings
    hiding: HidingSettings
    skeleton: SkeletonSettings
    panel: PanelSettings
    window: WindowSettings
    output: OutputSettings
    labels: LabelSettings


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


@dataclass(slots=True)
class ViewSettings:
    hide_style: HideStyle
    draw_skeleton: bool


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
