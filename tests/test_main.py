from collections.abc import Callable, Sequence

import pytest
from PySide6.QtWidgets import QApplication

from conftest import FakeCapture, FakePoseSource
from evidone_pose_detection import main as main_module
from evidone_pose_detection.camera import Camera
from evidone_pose_detection.errors import PoseModelError
from evidone_pose_detection.main import (
    CAMERA_EXIT_CODE,
    MODEL_EXIT_CODE,
    USAGE_EXIT_CODE,
    main,
    run,
)
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


def unavailable_model(config: PoseConfig) -> FakePoseSource:
    raise PoseModelError(config.weights)


@pytest.fixture
def built_applications(monkeypatch: pytest.MonkeyPatch) -> list[FakeApplication]:
    built: list[FakeApplication] = []

    def build(argv: Sequence[str]) -> FakeApplication:
        application = FakeApplication(argv, exit_code=0)
        built.append(application)
        return application

    monkeypatch.setattr(main_module, "QApplication", build)
    return built


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
    monkeypatch.setattr("sys.argv", ["evidone"])

    assert main() == exit_code
    assert working_capture.released == 1


def test_the_model_is_acquired_before_the_camera(
    install_capture: InstallCapture,
    built_applications: list[FakeApplication],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    opened: list[int] = install_capture(FakeCapture(frames=[], opened=True, fps=25.0))
    monkeypatch.setattr(main_module, "PoseEstimator", unavailable_model)
    monkeypatch.setattr("sys.argv", ["evidone"])

    assert main() == MODEL_EXIT_CODE
    assert opened == [], "the camera was opened even though the model had already failed"


def test_a_missing_model_reports_to_stderr_and_exits_non_zero(
    built_applications: list[FakeApplication],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(main_module, "PoseEstimator", unavailable_model)
    monkeypatch.setattr("sys.argv", ["evidone"])

    code = main()
    captured = capsys.readouterr()

    assert code == MODEL_EXIT_CODE
    assert code != 0
    assert captured.out == ""
    assert WEIGHTS in captured.err
    assert captured.err.count("\n") == 1, "the stderr report should be one short line"


def test_an_unavailable_camera_reports_to_stderr_and_exits_non_zero(
    install_capture: InstallCapture,
    built_applications: list[FakeApplication],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    install_capture(FakeCapture(frames=[], opened=False, fps=25.0))
    monkeypatch.setattr(main_module, "PoseEstimator", lambda config: FakePoseSource(()))
    monkeypatch.setattr("sys.argv", ["evidone", "--camera", "7"])

    code = main()
    captured = capsys.readouterr()

    assert code == CAMERA_EXIT_CODE
    assert code != 0
    assert captured.out == ""
    assert "7" in captured.err
    assert captured.err.count("\n") == 1, "the stderr report should be one short line"


@pytest.mark.parametrize("failing", ["model", "camera"])
def test_a_failed_start_never_builds_the_qt_application(
    install_capture: InstallCapture,
    built_applications: list[FakeApplication],
    monkeypatch: pytest.MonkeyPatch,
    failing: str,
) -> None:
    install_capture(FakeCapture(frames=[], opened=failing != "camera", fps=25.0))
    if failing == "model":
        monkeypatch.setattr(main_module, "PoseEstimator", unavailable_model)
    else:
        monkeypatch.setattr(main_module, "PoseEstimator", lambda config: FakePoseSource(()))
    monkeypatch.setattr("sys.argv", ["evidone"])

    assert main() != 0
    assert built_applications == []


def test_a_bad_argument_is_told_apart_from_a_missing_device() -> None:
    assert len({USAGE_EXIT_CODE, MODEL_EXIT_CODE, CAMERA_EXIT_CODE}) == 3
    assert USAGE_EXIT_CODE not in {MODEL_EXIT_CODE, CAMERA_EXIT_CODE}


@pytest.mark.parametrize("argv", [["evidone", "--camera", "front"], ["evidone", "--nope"]])
def test_a_bad_argument_exits_with_the_usage_code(
    monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> None:
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(SystemExit) as exit_info:
        main()

    assert exit_info.value.code == USAGE_EXIT_CODE


def test_run_receives_an_already_opened_camera(
    application: QApplication,
    working_capture: FakeCapture,
    install_capture: InstallCapture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_capture(working_capture)
    seen: list[Camera] = []

    def record(camera: Camera, poses: object) -> int:
        seen.append(camera)
        assert camera.read() is not None, "run() was handed a camera that was not open"
        return 0

    monkeypatch.setattr(main_module, "PoseEstimator", lambda config: FakePoseSource(()))
    monkeypatch.setattr(main_module, "run", record)
    monkeypatch.setattr("sys.argv", ["evidone"])

    assert main() == 0
    assert len(seen) == 1
    assert working_capture.released == 1


def test_run_builds_the_window_once_both_are_in_hand(
    application: QApplication,
    working_capture: FakeCapture,
    install_capture: InstallCapture,
    built_applications: list[FakeApplication],
) -> None:
    install_capture(working_capture)
    camera = Camera(CameraConfig(device_index=0))
    camera.open()

    assert run(camera, FakePoseSource(())) == 0
    assert len(built_applications) == 1
