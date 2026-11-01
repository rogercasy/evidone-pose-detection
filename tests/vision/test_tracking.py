import pytest

from evidone_pose_detection.models import FaceRegion
from evidone_pose_detection.vision.tracking import (
    HOLD_GROWTH_PER_FRAME,
    MATCH_DISTANCE_RATIO,
    SMOOTHING,
    FaceTracker,
)

RADIUS = 100.0
ALLOWANCE = RADIUS * MATCH_DISTANCE_RATIO

# Matching and smoothing are exercised without holding; holding has its own section.
NO_HOLDING = 0


def region(centre_x: float, centre_y: float, radius: float) -> FaceRegion:
    return FaceRegion(centre_x=centre_x, centre_y=centre_y, radius=radius)


def test_a_face_with_no_history_is_used_exactly_as_it_comes() -> None:
    measured = region(100.0, 100.0, RADIUS)

    (tracked,) = FaceTracker(hold_frames=NO_HOLDING).track([measured])

    assert tracked == measured


def test_a_matched_face_is_pulled_only_part_of_the_way() -> None:
    tracker = FaceTracker(hold_frames=NO_HOLDING)
    tracker.track([region(100.0, 100.0, RADIUS)])

    (tracked,) = tracker.track([region(120.0, 100.0, RADIUS)])

    assert tracked.centre_x == pytest.approx(100.0 + 20.0 * SMOOTHING)
    assert 100.0 < tracked.centre_x < 120.0


def test_the_size_is_smoothed_as_well_as_the_position() -> None:
    tracker = FaceTracker(hold_frames=NO_HOLDING)
    tracker.track([region(100.0, 100.0, RADIUS)])

    (tracked,) = tracker.track([region(100.0, 100.0, RADIUS + 40.0)])

    assert tracked.radius == pytest.approx(RADIUS + 40.0 * SMOOTHING)


def test_a_face_that_stops_moving_settles_on_the_measurement() -> None:
    tracker = FaceTracker(hold_frames=NO_HOLDING)
    tracker.track([region(100.0, 100.0, RADIUS)])
    settled = region(130.0, 100.0, RADIUS)

    for _ in range(40):
        (tracked,) = tracker.track([settled])

    assert tracked.centre_x == pytest.approx(130.0, abs=0.5)


def test_jitter_is_damped_rather_than_followed() -> None:
    tracker = FaceTracker(hold_frames=NO_HOLDING)
    steady = region(200.0, 200.0, RADIUS)
    for _ in range(20):
        tracker.track([steady])

    jittered = region(206.0, 200.0, RADIUS)
    (tracked,) = tracker.track([jittered])

    travelled = abs(tracked.centre_x - 200.0)
    assert travelled < 6.0, "the tracker followed the jitter one for one"
    assert travelled == pytest.approx(6.0 * SMOOTHING)


def test_a_face_that_jumped_too_far_starts_again_from_the_measurement() -> None:
    tracker = FaceTracker(hold_frames=NO_HOLDING)
    tracker.track([region(100.0, 100.0, RADIUS)])
    far = region(100.0 + ALLOWANCE + 1.0, 100.0, RADIUS)

    (tracked,) = tracker.track([far])

    assert tracked == far, "a rejected match should be used as it comes, not smoothed"


def test_a_face_at_the_limit_of_the_allowance_still_matches() -> None:
    tracker = FaceTracker(hold_frames=NO_HOLDING)
    tracker.track([region(100.0, 100.0, RADIUS)])

    (tracked,) = tracker.track([region(100.0 + ALLOWANCE, 100.0, RADIUS)])

    assert tracked.centre_x == pytest.approx(100.0 + ALLOWANCE * SMOOTHING)


@pytest.mark.parametrize(
    ("radius", "matched"), [(200.0, True), (100.0, True), (20.0, False), (5.0, False)]
)
def test_the_allowance_scales_with_the_size_of_the_face(radius: float, matched: bool) -> None:
    travel = 30.0
    tracker = FaceTracker(hold_frames=NO_HOLDING)
    tracker.track([region(100.0, 100.0, radius)])

    (tracked,) = tracker.track([region(100.0 + travel, 100.0, radius)])

    moved_the_whole_way = tracked.centre_x == pytest.approx(100.0 + travel)
    assert moved_the_whole_way is not matched


def test_a_previous_face_can_only_be_claimed_once() -> None:
    tracker = FaceTracker(hold_frames=NO_HOLDING)
    tracker.track([region(100.0, 100.0, RADIUS)])

    near = region(110.0, 100.0, RADIUS)
    nearer_still = region(105.0, 100.0, RADIUS)
    first, second = tracker.track([near, nearer_still])

    # The closer of the two takes the track; the other is left to start its own.
    assert second.centre_x == pytest.approx(100.0 + 5.0 * SMOOTHING)
    assert first == near


def test_each_face_matches_its_own_nearest_track() -> None:
    tracker = FaceTracker(hold_frames=NO_HOLDING)
    tracker.track([region(100.0, 100.0, RADIUS), region(400.0, 100.0, RADIUS)])

    left, right = tracker.track([region(110.0, 100.0, RADIUS), region(410.0, 100.0, RADIUS)])

    assert left.centre_x == pytest.approx(100.0 + 10.0 * SMOOTHING)
    assert right.centre_x == pytest.approx(400.0 + 10.0 * SMOOTHING)


def test_the_order_faces_arrive_in_does_not_change_the_pairing() -> None:
    previous = [region(100.0, 100.0, RADIUS), region(400.0, 100.0, RADIUS)]
    measured = [region(110.0, 100.0, RADIUS), region(410.0, 100.0, RADIUS)]

    forward = FaceTracker(hold_frames=NO_HOLDING)
    forward.track(previous)
    left, right = forward.track(measured)

    backward = FaceTracker(hold_frames=NO_HOLDING)
    backward.track(previous)
    right_first, left_second = backward.track(list(reversed(measured)))

    assert left_second.centre_x == pytest.approx(left.centre_x)
    assert right_first.centre_x == pytest.approx(right.centre_x)


def test_a_new_face_appearing_beside_a_tracked_one_is_used_as_it_comes() -> None:
    tracker = FaceTracker(hold_frames=NO_HOLDING)
    tracker.track([region(100.0, 100.0, RADIUS)])
    newcomer = region(900.0, 500.0, RADIUS)

    tracked, fresh = tracker.track([region(105.0, 100.0, RADIUS), newcomer])

    assert tracked.centre_x == pytest.approx(100.0 + 5.0 * SMOOTHING)
    assert fresh == newcomer


def test_an_empty_frame_gives_an_empty_result() -> None:
    tracker = FaceTracker(hold_frames=NO_HOLDING)
    tracker.track([region(100.0, 100.0, RADIUS)])

    assert tracker.track([]) == ()


def test_reset_forgets_the_history() -> None:
    tracker = FaceTracker(hold_frames=NO_HOLDING)
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
        tracker = FaceTracker(hold_frames=NO_HOLDING)
        return [tracker.track(frame) for frame in frames]

    assert run() == run()


# --- holding a face the model lost ------------------------------------------------------

HOLD = 3


def held_tracker() -> FaceTracker:
    return FaceTracker(hold_frames=HOLD)


def test_a_lost_face_stays_covered_for_the_configured_number_of_frames() -> None:
    tracker = held_tracker()
    tracker.track([region(100.0, 100.0, RADIUS)])

    covered = [len(tracker.track([])) for _ in range(HOLD + 2)]

    assert covered == [1] * HOLD + [0, 0]


@pytest.mark.parametrize("budget", [0, 1, 2, 5, 10])
def test_the_budget_decides_how_long_a_face_is_held(budget: int) -> None:
    tracker = FaceTracker(hold_frames=budget)
    tracker.track([region(100.0, 100.0, RADIUS)])

    frames_covered = 0
    while tracker.track([]):
        frames_covered += 1
        assert frames_covered <= budget + 1, "the face was never let go"

    assert frames_covered == budget


def test_a_face_with_no_budget_is_dropped_the_moment_it_is_lost() -> None:
    tracker = FaceTracker(hold_frames=0)
    tracker.track([region(100.0, 100.0, RADIUS)])

    assert tracker.track([]) == ()


def test_a_held_face_keeps_the_last_place_it_was_actually_seen() -> None:
    tracker = held_tracker()
    tracker.track([region(100.0, 250.0, RADIUS)])

    (held,) = tracker.track([])

    assert (held.centre_x, held.centre_y) == (100.0, 250.0)


def test_a_held_face_widens_because_we_no_longer_know_where_it_is() -> None:
    tracker = held_tracker()
    tracker.track([region(100.0, 100.0, RADIUS)])

    radii = []
    for _ in range(HOLD):
        (held,) = tracker.track([])
        radii.append(held.radius)

    assert radii[0] == pytest.approx(RADIUS * (1.0 + HOLD_GROWTH_PER_FRAME))
    assert radii == sorted(radii), "the cover should not shrink while the face is lost"
    assert radii[-1] > RADIUS


def test_a_face_found_again_resets_its_budget() -> None:
    tracker = held_tracker()
    tracker.track([region(100.0, 100.0, RADIUS)])
    for _ in range(HOLD - 1):
        tracker.track([])

    tracker.track([region(100.0, 100.0, RADIUS)])
    covered = [len(tracker.track([])) for _ in range(HOLD + 1)]

    assert covered == [1] * HOLD + [0], "the budget was not reset by finding the face again"


def test_a_held_face_can_be_picked_up_again_and_is_smoothed_from_where_it_was_held() -> None:
    tracker = held_tracker()
    tracker.track([region(100.0, 100.0, RADIUS)])
    tracker.track([])

    (found,) = tracker.track([region(110.0, 100.0, RADIUS)])

    assert found.centre_x == pytest.approx(100.0 + 10.0 * SMOOTHING)
    assert found.centre_x != 110.0


def test_holding_one_face_does_not_disturb_another_that_is_still_seen() -> None:
    tracker = held_tracker()
    tracker.track([region(100.0, 100.0, RADIUS), region(500.0, 100.0, RADIUS)])

    seen, held = tracker.track([region(105.0, 100.0, RADIUS)])

    assert seen.centre_x == pytest.approx(100.0 + 5.0 * SMOOTHING)
    assert held.centre_x == 500.0


def test_a_held_face_does_not_multiply_while_it_waits() -> None:
    tracker = held_tracker()
    tracker.track([region(100.0, 100.0, RADIUS)])

    for _ in range(HOLD):
        assert len(tracker.track([])) == 1


def test_reset_lets_go_of_held_faces_immediately() -> None:
    tracker = held_tracker()
    tracker.track([region(100.0, 100.0, RADIUS)])
    tracker.track([])

    tracker.reset()

    assert tracker.track([]) == ()


def test_holding_is_deterministic() -> None:
    frames: list[list[FaceRegion]] = [
        [region(100.0, 100.0, RADIUS), region(400.0, 100.0, RADIUS)],
        [region(104.0, 101.0, RADIUS)],
        [],
        [region(108.0, 102.0, RADIUS)],
        [],
    ]

    def run() -> list[tuple[FaceRegion, ...]]:
        tracker = held_tracker()
        return [tracker.track(frame) for frame in frames]

    assert run() == run()
