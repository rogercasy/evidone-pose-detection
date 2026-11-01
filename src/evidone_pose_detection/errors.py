class EvidoneError(Exception):
    pass


class CameraOpenError(EvidoneError):
    def __init__(self, device_index: int) -> None:
        super().__init__(f"Could not open camera device {device_index}")
        self.device_index = device_index


class CameraNotOpenError(EvidoneError):
    def __init__(self) -> None:
        super().__init__("The camera is not open")
