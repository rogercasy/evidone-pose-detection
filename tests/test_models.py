import pytest

from evidone_pose_detection.models import CaptureProperties


@pytest.mark.parametrize(("fps", "expected"), [(30.0, 33), (25.0, 40), (1.0, 1000), (60.0, 17)])
def test_frame_interval_follows_the_capture_rate(fps: float, expected: int) -> None:
    properties = CaptureProperties(width=640, height=480, fps=fps)

    assert properties.frame_interval_ms == expected
