import pytest

from evidone_pose_detection.ui.framerate import FrameRate


def test_the_first_frame_has_no_interval_to_measure() -> None:
    assert FrameRate(window=5).tick(100.0) == 0.0


def test_a_steady_cadence_reports_its_rate() -> None:
    meter = FrameRate(window=10)

    for index in range(6):
        rate = meter.tick(index * 0.04)

    assert rate == pytest.approx(25.0)


@pytest.mark.parametrize(("interval", "expected"), [(0.02, 50.0), (0.05, 20.0), (0.1, 10.0)])
def test_the_rate_is_the_inverse_of_the_interval(interval: float, expected: float) -> None:
    meter = FrameRate(window=10)

    for index in range(5):
        rate = meter.tick(index * interval)

    assert rate == pytest.approx(expected)


def test_the_rate_is_averaged_so_a_single_slow_frame_does_not_dominate() -> None:
    meter = FrameRate(window=10)
    now = 0.0
    for _ in range(9):
        now += 0.02
        meter.tick(now)

    now += 0.5
    rate = meter.tick(now)

    assert 10.0 < rate < 50.0


def test_only_the_most_recent_frames_count() -> None:
    meter = FrameRate(window=3)
    now = 0.0
    for _ in range(3):
        now += 1.0
        meter.tick(now)

    for _ in range(3):
        now += 0.02
        rate = meter.tick(now)

    assert rate == pytest.approx(50.0)


def test_two_frames_at_the_same_instant_do_not_divide_by_zero() -> None:
    meter = FrameRate(window=5)
    meter.tick(1.0)

    assert meter.tick(1.0) == 0.0
