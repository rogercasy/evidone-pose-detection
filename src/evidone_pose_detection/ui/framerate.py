from collections import deque


class FrameRate:
    def __init__(self, window: int) -> None:
        self._intervals: deque[float] = deque(maxlen=window)
        self._previous: float | None = None

    def tick(self, now: float) -> float:
        elapsed = self._elapsed_since_previous(now)
        if elapsed is not None and elapsed > 0.0:
            self._intervals.append(elapsed)
        if not self._intervals:
            return 0.0
        return len(self._intervals) / sum(self._intervals)

    def _elapsed_since_previous(self, now: float) -> float | None:
        previous, self._previous = self._previous, now
        if previous is None:
            return None
        return now - previous
