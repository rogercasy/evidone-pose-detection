from evidone_pose_detection.config import CONFIG
from evidone_pose_detection.models import Keypoint


def is_visible(keypoint: Keypoint) -> bool:
    return keypoint.confidence >= CONFIG.faces.visibility_threshold
