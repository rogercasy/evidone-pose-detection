from collections.abc import Callable, Sequence

import pytest
from PySide6.QtWidgets import QApplication

from conftest import FakeCapture, FakePoseSource
from evidone_pose_detection import main as main_module
from evidone_pose_detection.errors import CameraOpenError, PoseModelError
from evidone_pose_detection.main import STARTUP_ERROR_EXIT_CODE, main, run
from evidone_pose_detection.models import AppConfig, CameraConfig, PoseConfig

InstallCapture = Callable[[FakeCapture], list[int]]
WEIGHTS = "yolo26n-pose.pt"


class FakeApplication:
    def __init__(self, argv: Sequence[str], exit_code: int) -> None:
        self.argv = argv
        self._exit_code = exit_code

    def exec(self) -> int:
        return self._exit_code


def config_for(device_index: int) -> AppConfig:
    return AppConfig(
        camera=CameraConfig(device_index=device_index),
        pose=PoseConfig(weights=WEIGHTS, confidence=0.4),
    )


def raise_camera_open_error(config: AppConfig) -> int:
    raise CameraOpenError(config.camera.device_index)


def raise_pose_model_error(config: AppConfig) -> int:
    raise PoseModelError(config.pose.weights)


@pytest.mark.parametrize("exit_code", [0, 1])
def test_run_releases_the_device_once_the_event_loop_ends(
    application: QApplication,
    working_capture: FakeCapture,
    install_capture: InstallCapture,
    monkeypatch: pytest.MonkeyPatch,
    exit_code: int,
) -> None:
    install_capture(working_capture)
    monkeypatch.setattr(
        main_module, "QApplication", lambda argv: FakeApplication(argv, exit_code=exit_code)
    )
    monkeypatch.setattr(main_module, "PoseEstimator", lambda config: FakePoseSource(()))

    assert run(config_for(0)) == exit_code
    assert working_capture.released == 1


def test_run_loads_the_pose_model_before_the_event_loop_starts(
    application: QApplication,
    working_capture: FakeCapture,
    install_capture: InstallCapture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_capture(working_capture)
    loaded: list[PoseConfig] = []

    def load(config: PoseConfig) -> FakePoseSource:
        loaded.append(config)
        return FakePoseSource(())

    monkeypatch.setattr(main_module, "QApplication", lambda argv: FakeApplication(argv, 0))
    monkeypatch.setattr(main_module, "PoseEstimator", load)

    run(config_for(0))

    assert loaded == [PoseConfig(weights=WEIGHTS, confidence=0.4)]


@pytest.mark.parametrize(
    ("argv", "expected"),
    [(["evidone"], config_for(0)), (["evidone", "--camera", "2"], config_for(2))],
)
def test_main_passes_the_parsed_configuration_to_the_video_loop(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], expected: AppConfig
) -> None:
    opened: list[AppConfig] = []

    def record(config: AppConfig) -> int:
        opened.append(config)
        return 0

    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setattr(main_module, "run", record)

    assert main() == 0
    assert opened == [expected]


@pytest.mark.parametrize("failure", [raise_camera_open_error, raise_pose_model_error])
def test_main_reports_a_failed_startup_instead_of_crashing(
    monkeypatch: pytest.MonkeyPatch, failure: Callable[[AppConfig], int]
) -> None:
    monkeypatch.setattr("sys.argv", ["evidone"])
    monkeypatch.setattr(main_module, "run", failure)

    assert main() == STARTUP_ERROR_EXIT_CODE
