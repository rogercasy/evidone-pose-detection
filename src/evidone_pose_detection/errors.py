class EvidoneError(Exception):
    pass


class CameraOpenError(EvidoneError):
    def __init__(self, device_index: int) -> None:
        super().__init__(f"Could not open camera device {device_index}")
        self.device_index = device_index


class CameraNotOpenError(EvidoneError):
    def __init__(self) -> None:
        super().__init__("The camera is not open")


class PoseModelError(EvidoneError):
    def __init__(self, weights: str) -> None:
        super().__init__(f"Could not load the pose model {weights!r}")
        self.weights = weights


class UnexpectedKeypointCountError(EvidoneError):
    def __init__(self, count: int, expected: int) -> None:
        super().__init__(f"The model returned {count} keypoints per person, expected {expected}")
        self.count = count
        self.expected = expected
