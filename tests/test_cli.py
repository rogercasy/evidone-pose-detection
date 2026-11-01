import pytest

from evidone_pose_detection.cli import (
    DEFAULT_CAMERA_INDEX,
    DEFAULT_CONFIDENCE,
    DEFAULT_WEIGHTS,
    parse_config,
)


@pytest.mark.parametrize(
    ("argv", "expected"),
    [([], DEFAULT_CAMERA_INDEX), (["--camera", "0"], 0), (["--camera", "3"], 3)],
)
def test_parse_config_reads_the_device_index(argv: list[str], expected: int) -> None:
    assert parse_config(argv).camera.source == expected


@pytest.mark.parametrize(
    ("argv", "expected"),
    [([], DEFAULT_CONFIDENCE), (["--confidence", "0.1"], 0.1), (["--confidence", "1"], 1.0)],
)
def test_parse_config_reads_the_detection_threshold(argv: list[str], expected: float) -> None:
    assert parse_config(argv).pose.confidence == pytest.approx(expected)


def test_the_default_threshold_is_four_tenths() -> None:
    assert DEFAULT_CONFIDENCE == 0.4


def test_parse_config_names_the_pose_weights() -> None:
    assert parse_config([]).pose.weights == DEFAULT_WEIGHTS


@pytest.mark.parametrize(
    "argv",
    [
        ["--camera"],
        ["--camera", "front"],
        ["--unknown"],
        ["--confidence"],
        ["--confidence", "high"],
        ["--confidence", "0"],
        ["--confidence", "-0.2"],
        ["--confidence", "1.5"],
    ],
)
def test_parse_config_rejects_invalid_arguments(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse_config(argv)
