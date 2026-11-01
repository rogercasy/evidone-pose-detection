import logging
import sys

from PySide6.QtWidgets import QApplication

from evidone_pose_detection.camera import Camera
from evidone_pose_detection.cli import parse_config
from evidone_pose_detection.errors import CameraOpenError
from evidone_pose_detection.models import CameraConfig
from evidone_pose_detection.window import VideoWindow

logger = logging.getLogger(__name__)

CAMERA_ERROR_EXIT_CODE = 2


def main() -> int:
    config = parse_config(sys.argv[1:])
    try:
        return run(config)
    except CameraOpenError:
        logger.exception("Cannot start the video loop")
        return CAMERA_ERROR_EXIT_CODE


def run(config: CameraConfig) -> int:
    application = QApplication([])
    with Camera(config) as camera:
        window = VideoWindow(camera)
        window.show()
        window.start()
        return application.exec()


if __name__ == "__main__":
    sys.exit(main())
