import math
from collections.abc import Sequence
from dataclasses import dataclass

from evidone_pose_detection.models import FaceRegion

MATCH_DISTANCE_RATIO = 0.6
SMOOTHING = 0.3
DEFAULT_HOLD_FRAMES = 3
HOLD_GROWTH_PER_FRAME = 0.12


@dataclass(frozen=True, slots=True)
class _Track:
    region: FaceRegion
    held_frames: int


class FaceTracker:
    def __init__(self, hold_frames: int) -> None:
        self._hold_frames = hold_frames
        self._tracks: tuple[_Track, ...] = ()

    def track(self, regions: Sequence[FaceRegion]) -> tuple[FaceRegion, ...]:
        matches = _match(regions, [track.region for track in self._tracks])
        self._tracks = tuple(self._seen(regions, matches) + self._still_held(set(matches.values())))
        return tuple(track.region for track in self._tracks)

    def reset(self) -> None:
        self._tracks = ()

    def _seen(self, regions: Sequence[FaceRegion], matches: dict[int, int]) -> list[_Track]:
        return [
            _Track(_smoothed(self._tracks[matches[index]].region, region), held_frames=0)
            if index in matches
            else _Track(region, held_frames=0)
            for index, region in enumerate(regions)
        ]

    def _still_held(self, claimed: set[int]) -> list[_Track]:
        return [
            _held(track)
            for position, track in enumerate(self._tracks)
            if position not in claimed and track.held_frames < self._hold_frames
        ]


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


def _held(track: _Track) -> _Track:
    widened = FaceRegion(
        centre_x=track.region.centre_x,
        centre_y=track.region.centre_y,
        radius=track.region.radius * (1.0 + HOLD_GROWTH_PER_FRAME),
    )
    return _Track(widened, held_frames=track.held_frames + 1)


def _smoothed(previous: FaceRegion, measured: FaceRegion) -> FaceRegion:
    return FaceRegion(
        centre_x=_towards(previous.centre_x, measured.centre_x),
        centre_y=_towards(previous.centre_y, measured.centre_y),
        radius=_towards(previous.radius, measured.radius),
    )


def _towards(previous: float, measured: float) -> float:
    return previous + (measured - previous) * SMOOTHING
