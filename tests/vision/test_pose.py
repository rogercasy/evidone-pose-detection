from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from evidone_pose_detection.errors import PoseModelError, UnexpectedKeypointCountError
from evidone_pose_detection.models import KEYPOINT_COUNT, COCOKeypoint, Frame, PoseConfig
from evidone_pose_detection.vision import pose as pose_module
from evidone_pose_detection.vision.pose import PoseEstimator

WEIGHTS = "yolo26n-pose.pt"
CONFIDENCE = 0.4


class FakeBoxes:
    def __init__(self, xyxy: Any, conf: Any) -> None:
        self.xyxy = xyxy
        self.conf = conf


class FakeKeypoints:
    def __init__(self, xy: Any, conf: Any) -> None:
        self.xy = xy
        self.conf = conf


SPEED = {"preprocess": 1.25, "inference": 18.5, "postprocess": 0.75}


class FakeResult:
    def __init__(self, boxes: Any, keypoints: Any) -> None:
        self.boxes = boxes
        self.keypoints = keypoints
        self.speed = dict(SPEED)


class FakeYOLO:
    def __init__(self) -> None:
        self.results: list[Any] = []
        self.thresholds: list[float] = []

    def predict(self, frame: Frame, conf: float, verbose: bool) -> list[Any]:
        self.thresholds.append(conf)
        return self.results


def result_for(people: int, keypoints: int) -> FakeResult:
    return FakeResult(
        boxes=FakeBoxes(
            xyxy=np.array([[10.0, 20.0, 110.0, 220.0]] * people).reshape(people, 4),
            conf=np.array([0.77] * people),
        ),
        keypoints=FakeKeypoints(
            xy=np.array([[[float(i), float(i * 2)] for i in range(keypoints)]] * people).reshape(
                people, keypoints, 2
            ),
            conf=np.array([[0.5] * keypoints] * people).reshape(people, keypoints),
        ),
    )


@pytest.fixture
def model() -> FakeYOLO:
    return FakeYOLO()


@pytest.fixture
def loaded_weights(monkeypatch: pytest.MonkeyPatch, model: FakeYOLO) -> list[str]:
    weights: list[str] = []

    def build(name: str) -> FakeYOLO:
        weights.append(name)
        return model

    monkeypatch.setattr(pose_module, "YOLO", build)
    return weights


@pytest.fixture
def estimator(loaded_weights: list[str]) -> PoseEstimator:
    return PoseEstimator(PoseConfig(weights=WEIGHTS, confidence=CONFIDENCE))


def test_the_model_loads_once_at_startup_not_per_frame(
    estimator: PoseEstimator, model: FakeYOLO, loaded_weights: list[str], frame: Frame
) -> None:
    model.results = [result_for(people=1, keypoints=KEYPOINT_COUNT)]

    for _ in range(5):
        estimator.estimate(frame)

    assert loaded_weights == [WEIGHTS]
    assert len(model.thresholds) == 5


def test_estimate_passes_the_configured_threshold(
    estimator: PoseEstimator, model: FakeYOLO, frame: Frame
) -> None:
    estimator.estimate(frame)

    assert model.thresholds == [CONFIDENCE]


def test_a_detection_becomes_our_dataclasses(
    estimator: PoseEstimator, model: FakeYOLO, frame: Frame
) -> None:
    model.results = [result_for(people=1, keypoints=KEYPOINT_COUNT)]

    (person,) = estimator.estimate(frame).people

    assert (person.box.left, person.box.top) == (10.0, 20.0)
    assert (person.box.right, person.box.bottom) == (110.0, 220.0)
    assert person.confidence == pytest.approx(0.77)
    assert len(person.keypoints) == KEYPOINT_COUNT
    assert person.keypoint(COCOKeypoint.NOSE) is person.keypoints[0]
    assert person.keypoint(COCOKeypoint.RIGHT_ANKLE).confidence == pytest.approx(0.5)
    assert person.keypoint(COCOKeypoint.LEFT_WRIST).y == pytest.approx(18.0)


@pytest.mark.parametrize("people", [0, 1, 3])
def test_every_detected_person_is_converted(
    estimator: PoseEstimator, model: FakeYOLO, frame: Frame, people: int
) -> None:
    model.results = [result_for(people=people, keypoints=KEYPOINT_COUNT)]

    assert len(estimator.estimate(frame).people) == people


@pytest.mark.parametrize(
    "result",
    [
        FakeResult(boxes=None, keypoints=None),
        FakeResult(boxes=FakeBoxes(xyxy=np.empty((0, 4)), conf=np.empty(0)), keypoints=None),
        FakeResult(boxes=None, keypoints=FakeKeypoints(xy=np.empty((0, 17, 2)), conf=None)),
    ],
)
def test_a_result_without_detections_yields_nothing(
    estimator: PoseEstimator, model: FakeYOLO, frame: Frame, result: FakeResult
) -> None:
    model.results = [result]

    assert estimator.estimate(frame).people == ()


def test_missing_keypoint_scores_default_to_zero(
    estimator: PoseEstimator, model: FakeYOLO, frame: Frame
) -> None:
    result = result_for(people=1, keypoints=KEYPOINT_COUNT)
    result.keypoints.conf = None
    model.results = [result]

    (person,) = estimator.estimate(frame).people

    assert [point.confidence for point in person.keypoints] == [0.0] * KEYPOINT_COUNT


@pytest.mark.parametrize("keypoints", [13, 21])
def test_a_non_coco_model_is_rejected(
    estimator: PoseEstimator, model: FakeYOLO, frame: Frame, keypoints: int
) -> None:
    model.results = [result_for(people=1, keypoints=keypoints)]

    with pytest.raises(UnexpectedKeypointCountError) as error:
        estimator.estimate(frame)

    assert error.value.expected == KEYPOINT_COUNT
    assert error.value.count == keypoints


def test_a_model_that_cannot_load_raises_our_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pose_module, "YOLO", _unavailable_model)

    with pytest.raises(PoseModelError) as error:
        PoseEstimator(PoseConfig(weights=WEIGHTS, confidence=CONFIDENCE))

    assert error.value.weights == WEIGHTS


@pytest.mark.parametrize(
    ("name", "index"),
    [(COCOKeypoint.NOSE, 0), (COCOKeypoint.LEFT_WRIST, 9), (COCOKeypoint.RIGHT_ANKLE, 16)],
)
def test_coco_keypoints_keep_the_standard_order(name: COCOKeypoint, index: int) -> None:
    assert name == index
    assert KEYPOINT_COUNT == 17


def _unavailable_model(weights: str) -> Sequence[object]:
    message = "no such file"
    raise FileNotFoundError(message)


def test_the_timing_comes_from_what_the_model_reports(
    estimator: PoseEstimator, model: FakeYOLO, frame: Frame
) -> None:
    model.results = [result_for(people=1, keypoints=KEYPOINT_COUNT)]

    timing = estimator.estimate(frame).timing

    assert timing.preprocess_ms == pytest.approx(SPEED["preprocess"])
    assert timing.inference_ms == pytest.approx(SPEED["inference"])
    assert timing.postprocess_ms == pytest.approx(SPEED["postprocess"])


def test_the_total_is_the_three_stages_added_up(
    estimator: PoseEstimator, model: FakeYOLO, frame: Frame
) -> None:
    model.results = [result_for(people=1, keypoints=KEYPOINT_COUNT)]

    timing = estimator.estimate(frame).timing

    assert timing.total_ms == pytest.approx(sum(SPEED.values()))
    assert timing.total_ms > timing.inference_ms


def test_a_stage_the_model_did_not_report_counts_as_zero(
    estimator: PoseEstimator, model: FakeYOLO, frame: Frame
) -> None:
    result = result_for(people=1, keypoints=KEYPOINT_COUNT)
    result.speed = {"inference": 9.0}
    model.results = [result]

    timing = estimator.estimate(frame).timing

    assert timing.inference_ms == pytest.approx(9.0)
    assert timing.preprocess_ms == 0.0
    assert timing.postprocess_ms == 0.0


def test_timing_without_any_result_is_zero(estimator: PoseEstimator, frame: Frame) -> None:
    timing = estimator.estimate(frame).timing

    assert timing.total_ms == 0.0
