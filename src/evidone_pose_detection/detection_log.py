import json
import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

from evidone_pose_detection.errors import DetectionLogError
from evidone_pose_detection.models import COCOKeypoint, PersonPose

logger = logging.getLogger(__name__)


class DetectionLog:
    def __init__(self) -> None:
        self._file: TextIO | None = None
        self._destination: Path | None = None
        self._frame = 0

    @property
    def is_logging(self) -> bool:
        return self._file is not None

    @property
    def destination(self) -> Path | None:
        return self._destination

    def start(self, destination: Path) -> None:
        self.close()
        try:
            # Line buffered, so every finished line reaches the file as it is written.
            self._file = destination.open("w", buffering=1, encoding="utf-8")
        except OSError as error:
            raise DetectionLogError(destination) from error
        self._destination = destination
        self._frame = 0
        logger.info("Logging detections to %s", destination)

    def write(self, people: Sequence[PersonPose]) -> None:
        if self._file is None:
            return
        # A frame with nobody in it still gets a line, so the numbering keeps step
        # with a recording of the same session.
        self._file.write(json.dumps(_record(self._frame, people)) + "\n")
        self._frame += 1

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            logger.info("Finished logging %d frames to %s", self._frame, self._destination)
        self._file = None
        self._destination = None


def _record(frame: int, people: Sequence[PersonPose]) -> dict[str, Any]:
    return {"frame": frame, "people": [_person(person) for person in people]}


def _person(person: PersonPose) -> dict[str, Any]:
    return {
        "score": person.confidence,
        "box": {
            "left": person.box.left,
            "top": person.box.top,
            "right": person.box.right,
            "bottom": person.box.bottom,
        },
        # Every keypoint the model produced, including the ones it was unsure about:
        # filtering is the reader's decision, not ours.
        "keypoints": [
            {
                "name": name.name.lower(),
                "x": keypoint.x,
                "y": keypoint.y,
                "confidence": keypoint.confidence,
            }
            for name, keypoint in zip(COCOKeypoint, person.keypoints, strict=True)
        ],
    }
