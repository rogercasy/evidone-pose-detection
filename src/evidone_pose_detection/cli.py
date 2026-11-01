import argparse
from collections.abc import Sequence

from evidone_pose_detection.config import CONFIG
from evidone_pose_detection.models import AppConfig, CameraConfig, PoseConfig


def parse_config(argv: Sequence[str]) -> AppConfig:
    parser = argparse.ArgumentParser(
        prog="evidone", description="Show a live camera feed with pose detection."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=CONFIG.camera.default_index,
        metavar="INDEX",
        help="index of the capture device to open (default: %(default)s)",
    )
    parser.add_argument(
        "--confidence",
        type=_confidence,
        default=CONFIG.pose.confidence,
        metavar="SCORE",
        help="minimum detection score, between 0 and 1 (default: %(default)s)",
    )
    arguments = parser.parse_args(argv)
    return AppConfig(
        camera=CameraConfig(source=arguments.camera),
        pose=PoseConfig(weights=CONFIG.pose.weights, confidence=arguments.confidence),
    )


def _confidence(value: str) -> float:
    score = float(value)
    if not 0.0 < score <= 1.0:
        message = f"must be greater than 0 and at most 1, got {value}"
        raise argparse.ArgumentTypeError(message)
    return score
