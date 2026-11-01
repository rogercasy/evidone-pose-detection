import pytest

from evidone_pose_detection.models import FaceRegion
from evidone_pose_detection.vision.tracking import MATCH_DISTANCE_RATIO, SMOOTHING, FaceTracker

RADIUS = 100.0
ALLOWANCE = RADIUS * MATCH_DISTANCE_RATIO


def region(centre_x: float, centre_y: float, radius: float) -> FaceRegion:
    return FaceRegion(centre_x=centre_x, centre_y=centre_y, radius=radius)


def test_a_face_with_no_history_is_used_exactly_as_it_comes() -> None:
    measured = region(100.0, 100.0, RADIUS)

    (tracked,) = FaceTracker().track([measured])

    assert tracked == measured


def test_a_matched_face_is_pulled_only_part_of_the_way() -> None:
    tracker = FaceTracker()
    tracker.track([region(100.0, 100.0, RADIUS)])

    (tracked,) = tracker.track([region(120.0, 100.0, RADIUS)])

    assert tracked.centre_x == pytest.approx(100.0 + 20.0 * SMOOTHING)
    assert 100.0 < tracked.centre_x < 120.0


def test_the_size_is_smoothed_as_well_as_the_position() -> None:
    tracker = FaceTracker()
    tracker.track([region(100.0, 100.0, RADIUS)])

    (tracked,) = tracker.track([region(100.0, 100.0, RADIUS + 40.0)])

    assert tracked.radius == pytest.approx(RADIUS + 40.0 * SMOOTHING)


def test_a_face_that_stops_moving_settles_on_the_measurement() -> None:
    tracker = FaceTracker()
    tracker.track([region(100.0, 100.0, RADIUS)])
    settled = region(130.0, 100.0, RADIUS)

    for _ in range(40):
        (tracked,) = tracker.track([settled])

    assert tracked.centre_x == pytest.approx(130.0, abs=0.5)


def test_jitter_is_damped_rather_than_followed() -> None:
    tracker = FaceTracker()
    steady = region(200.0, 200.0, RADIUS)
    for _ in range(20):
        tracker.track([steady])

    jittered = region(206.0, 200.0, RADIUS)
    (tracked,) = tracker.track([jittered])

    travelled = abs(tracked.centre_x - 200.0)
    assert travelled < 6.0, "the tracker followed the jitter one for one"
    assert travelled == pytest.approx(6.0 * SMOOTHING)


def test_a_face_that_jumped_too_far_starts_again_from_the_measurement() -> None:
    tracker = FaceTracker()
    tracker.track([region(100.0, 100.0, RADIUS)])
    far = region(100.0 + ALLOWANCE + 1.0, 100.0, RADIUS)

    (tracked,) = tracker.track([far])

    assert tracked == far, "a rejected match should be used as it comes, not smoothed"


def test_a_face_at_the_limit_of_the_allowance_still_matches() -> None:
    tracker = FaceTracker()
    tracker.track([region(100.0, 100.0, RADIUS)])

    (tracked,) = tracker.track([region(100.0 + ALLOWANCE, 100.0, RADIUS)])

    assert tracked.centre_x == pytest.approx(100.0 + ALLOWANCE * SMOOTHING)


@pytest.mark.parametrize(
    ("radius", "matched"), [(200.0, True), (100.0, True), (20.0, False), (5.0, False)]
)
def test_the_allowance_scales_with_the_size_of_the_face(radius: float, matched: bool) -> None:
    travel = 30.0
    tracker = FaceTracker()
    tracker.track([region(100.0, 100.0, radius)])

    (tracked,) = tracker.track([region(100.0 + travel, 100.0, radius)])

    moved_the_whole_way = tracked.centre_x == pytest.approx(100.0 + travel)
    assert moved_the_whole_way is not matched


def test_a_previous_face_can_only_be_claimed_once() -> None:
    tracker = FaceTracker()
    tracker.track([region(100.0, 100.0, RADIUS)])

    near = region(110.0, 100.0, RADIUS)
    nearer_still = region(105.0, 100.0, RADIUS)
    first, second = tracker.track([near, nearer_still])

    # The closer of the two takes the track; the other is left to start its own.
    assert second.centre_x == pytest.approx(100.0 + 5.0 * SMOOTHING)
    assert first == near


def test_each_face_matches_its_own_nearest_track() -> None:
    tracker = FaceTracker()
    tracker.track([region(100.0, 100.0, RADIUS), region(400.0, 100.0, RADIUS)])

    left, right = tracker.track([region(110.0, 100.0, RADIUS), region(410.0, 100.0, RADIUS)])

    assert left.centre_x == pytest.approx(100.0 + 10.0 * SMOOTHING)
    assert right.centre_x == pytest.approx(400.0 + 10.0 * SMOOTHING)


def test_the_order_faces_arrive_in_does_not_change_the_pairing() -> None:
    previous = [region(100.0, 100.0, RADIUS), region(400.0, 100.0, RADIUS)]
    measured = [region(110.0, 100.0, RADIUS), region(410.0, 100.0, RADIUS)]

    forward = FaceTracker()
    forward.track(previous)
    left, right = forward.track(measured)

    backward = FaceTracker()
    backward.track(previous)
    right_first, left_second = backward.track(list(reversed(measured)))

    assert left_second.centre_x == pytest.approx(left.centre_x)
    assert right_first.centre_x == pytest.approx(right.centre_x)


def test_a_new_face_appearing_beside_a_tracked_one_is_used_as_it_comes() -> None:
    tracker = FaceTracker()
    tracker.track([region(100.0, 100.0, RADIUS)])
    newcomer = region(900.0, 500.0, RADIUS)

    tracked, fresh = tracker.track([region(105.0, 100.0, RADIUS), newcomer])

    assert tracked.centre_x == pytest.approx(100.0 + 5.0 * SMOOTHING)
    assert fresh == newcomer


def test_an_empty_frame_gives_an_empty_result() -> None:
    tracker = FaceTracker()
    tracker.track([region(100.0, 100.0, RADIUS)])

    assert tracker.track([]) == ()


def test_reset_forgets_the_history() -> None:
    tracker = FaceTracker()
    tracker.track([region(100.0, 100.0, RADIUS)])

    tracker.reset()
    measured = region(110.0, 100.0, RADIUS)
    (tracked,) = tracker.track([measured])

    assert tracked == measured


def test_the_same_inputs_always_give_the_same_outputs() -> None:
    frames = [
        [region(100.0, 100.0, RADIUS), region(400.0, 120.0, RADIUS)],
        [region(108.0, 103.0, RADIUS + 5.0), region(395.0, 118.0, RADIUS)],
        [region(115.0, 99.0, RADIUS - 3.0)],
    ]

    def run() -> list[tuple[FaceRegion, ...]]:
        tracker = FaceTracker()
        return [tracker.track(frame) for frame in frames]

    assert run() == run()
