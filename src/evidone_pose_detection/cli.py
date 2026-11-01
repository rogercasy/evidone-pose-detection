import argparse
from collections.abc import Sequence

from evidone_pose_detection.models import AppConfig, CameraConfig, PoseConfig

DEFAULT_CAMERA_INDEX = 0
DEFAULT_CONFIDENCE = 0.4
DEFAULT_WEIGHTS = "yolo26n-pose.pt"


def parse_config(argv: Sequence[str]) -> AppConfig:
    parser = argparse.ArgumentParser(
        prog="evidone", description="Show a live camera feed with pose detection."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=DEFAULT_CAMERA_INDEX,
        metavar="INDEX",
        help="index of the capture device to open (default: %(default)s)",
    )
    parser.add_argument(
        "--confidence",
        type=_confidence,
        default=DEFAULT_CONFIDENCE,
        metavar="SCORE",
        help="minimum detection score, between 0 and 1 (default: %(default)s)",
    )
    arguments = parser.parse_args(argv)
    return AppConfig(
        camera=CameraConfig(source=arguments.camera),
        pose=PoseConfig(weights=DEFAULT_WEIGHTS, confidence=arguments.confidence),
    )


def _confidence(value: str) -> float:
    score = float(value)
    if not 0.0 < score <= 1.0:
        message = f"must be greater than 0 and at most 1, got {value}"
        raise argparse.ArgumentTypeError(message)
    return score
