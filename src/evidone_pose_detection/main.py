import logging
import sys

from PySide6.QtWidgets import QApplication

from evidone_pose_detection.camera import Camera
from evidone_pose_detection.cli import parse_config
from evidone_pose_detection.errors import CameraOpenError, EvidoneError, PoseModelError
from evidone_pose_detection.models import PoseSource
from evidone_pose_detection.ui import VideoWindow
from evidone_pose_detection.vision.pose import PoseEstimator

logger = logging.getLogger(__name__)

PROGRAM = "evidone"

USAGE_EXIT_CODE = 2
MODEL_EXIT_CODE = 3
CAMERA_EXIT_CODE = 4


def main() -> int:
    config = parse_config(sys.argv[1:])

    try:
        estimator = PoseEstimator(config.pose)
    except PoseModelError as error:
        return _abort(error, MODEL_EXIT_CODE)

    try:
        with Camera(config.camera) as camera:
            return run(camera, estimator)
    except CameraOpenError as error:
        return _abort(error, CAMERA_EXIT_CODE)


def run(camera: Camera, poses: PoseSource) -> int:
    application = QApplication([])
    window = VideoWindow(camera, poses)
    try:
        window.show()
        window.start()
        return application.exec()
    finally:
        window.shutdown()


def _abort(error: EvidoneError, code: int) -> int:
    logger.debug("Startup failed", exc_info=error)
    sys.stderr.write(f"{PROGRAM}: {error}\n")
    return code


if __name__ == "__main__":
    sys.exit(main())
