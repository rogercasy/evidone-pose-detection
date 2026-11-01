from pathlib import Path

from evidone_pose_detection.models import CaptureSource


class EvidoneError(Exception):
    pass


class ConfigError(EvidoneError):
    pass


class CameraOpenError(EvidoneError):
    def __init__(self, source: CaptureSource) -> None:
        super().__init__(f"could not open {_describe(source)}")
        self.source = source


class CameraNotOpenError(EvidoneError):
    def __init__(self) -> None:
        super().__init__("the camera is not open")


class PoseModelError(EvidoneError):
    def __init__(self, weights: str) -> None:
        super().__init__(
            f"could not load the pose model {weights!r}: "
            "the weights file may be missing, unreadable, or the download failed"
        )
        self.weights = weights


class RecordingError(EvidoneError):
    def __init__(self, destination: Path) -> None:
        super().__init__(
            f"could not record to '{destination}': "
            "the folder may not exist, or the codec may be unavailable"
        )
        self.destination = destination


class DetectionLogError(EvidoneError):
    def __init__(self, destination: Path) -> None:
        super().__init__(
            f"could not log detections to '{destination}': "
            "the folder may not exist, or it may not be writable"
        )
        self.destination = destination


class UnexpectedKeypointCountError(EvidoneError):
    def __init__(self, count: int, expected: int) -> None:
        super().__init__(f"the model returned {count} keypoints per person, expected {expected}")
        self.count = count
        self.expected = expected


def _describe(source: CaptureSource) -> str:
    if isinstance(source, Path):
        return (
            f"video file '{source}': "
            "it may be missing, unreadable, or in a format OpenCV cannot decode"
        )
    return (
        f"camera device {source}: "
        "it may be disconnected, in use by another application, or not a capture device"
    )
