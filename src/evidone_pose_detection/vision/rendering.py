from collections.abc import Sequence

import cv2

from evidone_pose_detection.config import CONFIG
from evidone_pose_detection.models import (
    KEYPOINT_REGIONS,
    SKELETON,
    COCOKeypoint,
    Frame,
    Keypoint,
    PersonPose,
)
from evidone_pose_detection.vision.visibility import is_visible


def draw_poses(frame: Frame, people: Sequence[PersonPose]) -> Frame:
    for person in people:
        _draw_bones(frame, person)
        _draw_joints(frame, person)
    return frame


def _draw_bones(frame: Frame, person: PersonPose) -> None:
    for bone in SKELETON:
        start = person.keypoint(bone.start)
        end = person.keypoint(bone.end)
        if is_visible(start) and is_visible(end):
            cv2.line(
                frame,
                _point(start),
                _point(end),
                CONFIG.skeleton.colours[bone.region],
                CONFIG.skeleton.bone_thickness,
                cv2.LINE_AA,
            )


def _draw_joints(frame: Frame, person: PersonPose) -> None:
    for name, keypoint in zip(COCOKeypoint, person.keypoints, strict=True):
        if is_visible(keypoint):
            cv2.circle(
                frame,
                _point(keypoint),
                CONFIG.skeleton.joint_radius,
                CONFIG.skeleton.colours[KEYPOINT_REGIONS[name]],
                cv2.FILLED,
                cv2.LINE_AA,
            )


def _point(keypoint: Keypoint) -> tuple[int, int]:
    return round(keypoint.x), round(keypoint.y)
