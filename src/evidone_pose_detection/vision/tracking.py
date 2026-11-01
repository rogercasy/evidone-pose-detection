import math
from collections.abc import Sequence

from evidone_pose_detection.models import FaceRegion

MATCH_DISTANCE_RATIO = 0.6

SMOOTHING = 0.3


class FaceTracker:
    def __init__(self) -> None:
        self._previous: tuple[FaceRegion, ...] = ()

    def track(self, regions: Sequence[FaceRegion]) -> tuple[FaceRegion, ...]:
        matches = _match(regions, self._previous)
        tracked = tuple(
            _smoothed(self._previous[matches[index]], region) if index in matches else region
            for index, region in enumerate(regions)
        )
        self._previous = tracked
        return tracked

    def reset(self) -> None:
        self._previous = ()


def _match(regions: Sequence[FaceRegion], previous: Sequence[FaceRegion]) -> dict[int, int]:
    claimed: set[int] = set()
    matches: dict[int, int] = {}
    for _, new_index, old_index in sorted(_candidates(regions, previous)):
        if new_index in matches or old_index in claimed:
            continue
        matches[new_index] = old_index
        claimed.add(old_index)
    return matches


def _candidates(
    regions: Sequence[FaceRegion], previous: Sequence[FaceRegion]
) -> list[tuple[float, int, int]]:
    near: list[tuple[float, int, int]] = []
    for new_index, region in enumerate(regions):
        for old_index, old in enumerate(previous):
            distance = math.hypot(region.centre_x - old.centre_x, region.centre_y - old.centre_y)
            if distance <= max(region.radius, old.radius) * MATCH_DISTANCE_RATIO:
                near.append((distance, new_index, old_index))
    return near


def _smoothed(previous: FaceRegion, measured: FaceRegion) -> FaceRegion:
    return FaceRegion(
        centre_x=_towards(previous.centre_x, measured.centre_x),
        centre_y=_towards(previous.centre_y, measured.centre_y),
        radius=_towards(previous.radius, measured.radius),
    )


def _towards(previous: float, measured: float) -> float:
    return previous + (measured - previous) * SMOOTHING
