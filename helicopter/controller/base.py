from abc import ABC, abstractmethod

class FlightController(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def control(self, **kwargs) -> float:
        # Output signals should always be on [-1, 1] or [0, 1].
        # It is the job of the aircraft to scale it correctly
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def format_command(command, trim=0, channel=0) -> list[int]:
        raise NotImplementedError
