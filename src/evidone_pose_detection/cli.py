import argparse
from collections.abc import Sequence

from evidone_pose_detection.models import CameraConfig

DEFAULT_CAMERA_INDEX = 0


def parse_config(argv: Sequence[str]) -> CameraConfig:
    parser = argparse.ArgumentParser(
        prog="evidone", description="Show a live camera feed in a window."
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=DEFAULT_CAMERA_INDEX,
        metavar="INDEX",
        help="index of the capture device to open (default: %(default)s)",
    )
    arguments = parser.parse_args(argv)
    return CameraConfig(device_index=arguments.camera)
