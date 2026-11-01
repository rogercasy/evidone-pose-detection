from collections.abc import Callable, Sequence

import pytest
from PySide6.QtWidgets import QApplication

from conftest import FakeCapture
from evidone_pose_detection import main as main_module
from evidone_pose_detection.errors import CameraOpenError
from evidone_pose_detection.main import CAMERA_ERROR_EXIT_CODE, main, run
from evidone_pose_detection.models import CameraConfig

InstallCapture = Callable[[FakeCapture], list[int]]


class FakeApplication:
    def __init__(self, argv: Sequence[str], exit_code: int) -> None:
        self.argv = argv
        self._exit_code = exit_code

    def exec(self) -> int:
        return self._exit_code


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

    assert run(CameraConfig(device_index=0)) == exit_code
    assert working_capture.released == 1


@pytest.mark.parametrize(
    ("argv", "expected"), [(["evidone"], 0), (["evidone", "--camera", "2"], 2)]
)
def test_main_passes_the_requested_device_to_the_video_loop(
    monkeypatch: pytest.MonkeyPatch, argv: list[str], expected: int
) -> None:
    opened: list[CameraConfig] = []

    def record(config: CameraConfig) -> int:
        opened.append(config)
        return 0

    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setattr(main_module, "run", record)

    assert main() == 0
    assert opened == [CameraConfig(device_index=expected)]


def test_main_reports_an_unavailable_camera_instead_of_crashing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.argv", ["evidone"])
    monkeypatch.setattr(main_module, "run", _raise_camera_open_error)

    assert main() == CAMERA_ERROR_EXIT_CODE


def _raise_camera_open_error(config: CameraConfig) -> int:
    raise CameraOpenError(config.device_index)
