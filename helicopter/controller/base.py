from abc import ABC, abstractmethod

import numpy as np

class FlightController(ABC):
    N: int = 3
    dtype = np.float64

    def __init__(self):
        self.killed = False

    @abstractmethod
    def control(self, **kwargs) -> np.ndarray:
        # Output signals should always be on [-1, 1] or [0, 1].
        # It is the job of the aircraft to scale it correctly
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def format_command(command, trim=0, channel=0) -> list[int]:
        raise NotImplementedError

    def shutdown(self):
        pass
