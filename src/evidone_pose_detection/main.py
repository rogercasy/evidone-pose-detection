import logging
import os

import evidone_pose_detection

logger = logging.getLogger(__name__)

COOKIE_PYRATE_VERSION = os.getenv("COOKIE_PYRATE_VERSION", "Not Set")


def main() -> None:
    pkg = "evidone_pose_detection"
    version = evidone_pose_detection.__version__

    logger.info("Package '%s' installed with version: %s", pkg, version)
    logger.info("Generated using Cookie Pyrate Version: %s", COOKIE_PYRATE_VERSION)

    logger.info("Life is beautiful!")


if __name__ == "__main__":
    main()
