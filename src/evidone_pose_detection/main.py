import logging
import sys

from PySide6.QtWidgets import QApplication

from evidone_pose_detection.camera import Camera
from evidone_pose_detection.cli import parse_config
from evidone_pose_detection.errors import CameraOpenError, PoseModelError
from evidone_pose_detection.models import AppConfig
from evidone_pose_detection.ui import VideoWindow
from evidone_pose_detection.vision.pose import PoseEstimator

logger = logging.getLogger(__name__)

STARTUP_ERROR_EXIT_CODE = 2


def main() -> int:
    config = parse_config(sys.argv[1:])
    try:
        return run(config)
    except (CameraOpenError, PoseModelError):
        logger.exception("Cannot start the video loop")
        return STARTUP_ERROR_EXIT_CODE


def run(config: AppConfig) -> int:
    application = QApplication([])
    estimator = PoseEstimator(config.pose)
    with Camera(config.camera) as camera:
        window = VideoWindow(camera, estimator)
        window.show()
        window.start()
        return application.exec()


if __name__ == "__main__":
    sys.exit(main())
