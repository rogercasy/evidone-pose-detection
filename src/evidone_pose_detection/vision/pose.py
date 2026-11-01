import logging
from collections.abc import Mapping, Sequence
from typing import cast

from ultralytics.engine.results import Keypoints, Results
from ultralytics.models.yolo.model import YOLO

from evidone_pose_detection.errors import PoseModelError, UnexpectedKeypointCountError
from evidone_pose_detection.models import (
    KEYPOINT_COUNT,
    BoundingBox,
    Frame,
    InferenceTiming,
    Keypoint,
    PersonPose,
    PoseConfig,
    PoseResult,
)

logger = logging.getLogger(__name__)


class PoseEstimator:
    def __init__(self, config: PoseConfig) -> None:
        self._config = config
        self._model = _load_model(config.weights)
        logger.info("Loaded pose model %s", config.weights)

    def estimate(self, frame: Frame) -> PoseResult:
        # predict() is typed for streaming too; without stream=True it is always list[Results].
        results = cast(
            list[Results], self._model.predict(frame, conf=self._config.confidence, verbose=False)
        )
        return PoseResult(
            people=tuple(person for result in results for person in _to_person_poses(result)),
            timing=_timing(results),
        )


def _timing(results: Sequence[Results]) -> InferenceTiming:
    speeds = [result.speed for result in results]
    return InferenceTiming(
        preprocess_ms=_total(speeds, "preprocess"),
        inference_ms=_total(speeds, "inference"),
        postprocess_ms=_total(speeds, "postprocess"),
    )


def _total(speeds: Sequence[Mapping[str, float | None]], stage: str) -> float:
    return sum(float(speed.get(stage) or 0.0) for speed in speeds)


def _load_model(weights: str) -> YOLO:
    try:
        return YOLO(weights)
    except Exception as error:
        raise PoseModelError(weights) from error


def _to_person_poses(result: Results) -> list[PersonPose]:
    boxes = result.boxes
    keypoints = result.keypoints
    if boxes is None or keypoints is None:
        return []

    positions = keypoints.xy.tolist()
    return [
        PersonPose(
            box=BoundingBox(left=box[0], top=box[1], right=box[2], bottom=box[3]),
            confidence=score,
            keypoints=_to_keypoints(person_positions, person_scores),
        )
        for box, score, person_positions, person_scores in zip(
            boxes.xyxy.tolist(),
            boxes.conf.tolist(),
            positions,
            _keypoint_scores(keypoints, len(positions)),
            strict=True,
        )
    ]


def _keypoint_scores(keypoints: Keypoints, people: int) -> list[list[float]]:
    if keypoints.conf is None:
        return [[0.0] * KEYPOINT_COUNT for _ in range(people)]
    return keypoints.conf.tolist()


def _to_keypoints(
    positions: Sequence[Sequence[float]], scores: Sequence[float]
) -> tuple[Keypoint, ...]:
    if len(positions) != KEYPOINT_COUNT:
        raise UnexpectedKeypointCountError(len(positions), KEYPOINT_COUNT)
    return tuple(
        Keypoint(x=x, y=y, confidence=score)
        for (x, y), score in zip(positions, scores, strict=True)
    )
