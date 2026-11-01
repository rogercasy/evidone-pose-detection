class EvidoneError(Exception):
    pass


class CameraOpenError(EvidoneError):
    def __init__(self, device_index: int) -> None:
        super().__init__(
            f"could not open camera device {device_index}: "
            "it may be disconnected, in use by another application, or not a capture device"
        )
        self.device_index = device_index


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


class UnexpectedKeypointCountError(EvidoneError):
    def __init__(self, count: int, expected: int) -> None:
        super().__init__(f"the model returned {count} keypoints per person, expected {expected}")
        self.count = count
        self.expected = expected
