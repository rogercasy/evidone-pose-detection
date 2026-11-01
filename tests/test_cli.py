import pytest

from evidone_pose_detection.cli import DEFAULT_CAMERA_INDEX, parse_config


@pytest.mark.parametrize(
    ("argv", "expected"),
    [([], DEFAULT_CAMERA_INDEX), (["--camera", "0"], 0), (["--camera", "3"], 3)],
)
def test_parse_config_reads_the_device_index(argv: list[str], expected: int) -> None:
    assert parse_config(argv).device_index == expected


@pytest.mark.parametrize("argv", [["--camera"], ["--camera", "front"], ["--unknown"]])
def test_parse_config_rejects_invalid_arguments(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse_config(argv)
