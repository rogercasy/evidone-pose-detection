from collections.abc import Mapping, Sequence
from types import MappingProxyType

import cv2

from evidone_pose_detection.models import (
    KEYPOINT_REGIONS,
    SKELETON,
    BodyRegion,
    COCOKeypoint,
    Frame,
    Keypoint,
    PersonPose,
)

VISIBILITY_THRESHOLD = 0.5

REGION_COLOURS: Mapping[BodyRegion, tuple[int, int, int]] = MappingProxyType(
    {
        BodyRegion.HEAD: (0, 195, 255),
        BodyRegion.TORSO: (255, 186, 59),
        BodyRegion.LIMB: (104, 222, 122),
    }
)

JOINT_RADIUS = 4
BONE_THICKNESS = 2


def draw_poses(frame: Frame, people: Sequence[PersonPose]) -> Frame:
    for person in people:
        _draw_bones(frame, person)
        _draw_joints(frame, person)
    return frame


def _draw_bones(frame: Frame, person: PersonPose) -> None:
    for bone in SKELETON:
        start = person.keypoint(bone.start)
        end = person.keypoint(bone.end)
        if _is_visible(start) and _is_visible(end):
            cv2.line(
                frame,
                _point(start),
                _point(end),
                REGION_COLOURS[bone.region],
                BONE_THICKNESS,
                cv2.LINE_AA,
            )


def _draw_joints(frame: Frame, person: PersonPose) -> None:
    for name, keypoint in zip(COCOKeypoint, person.keypoints, strict=True):
        if _is_visible(keypoint):
            cv2.circle(
                frame,
                _point(keypoint),
                JOINT_RADIUS,
                REGION_COLOURS[KEYPOINT_REGIONS[name]],
                cv2.FILLED,
                cv2.LINE_AA,
            )


def _is_visible(keypoint: Keypoint) -> bool:
    return keypoint.confidence >= VISIBILITY_THRESHOLD


def _point(keypoint: Keypoint) -> tuple[int, int]:
    return round(keypoint.x), round(keypoint.y)
