from dataclasses import dataclass

import cv2

type Frame = cv2.typing.MatLike


@dataclass(frozen=True, slots=True)
class CameraConfig:
    device_index: int


@dataclass(frozen=True, slots=True)
class CaptureProperties:
    width: int
    height: int
    fps: float

    @property
    def frame_interval_ms(self) -> int:
        return round(1000 / self.fps)
