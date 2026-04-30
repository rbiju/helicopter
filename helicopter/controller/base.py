from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.transform import Rotation

from helicopter.flightplan import FlightPlan


class FlightController(ABC):
    N: int = 3
    dtype = np.float64

    def __init__(self):
        self.killed = False

    @abstractmethod
    def control(self, flightplan: FlightPlan,
                quaternion: Rotation,
                position: np.ndarray,
                timestamp: float) -> np.ndarray:
        # Output signals should always be on [-1, 1] or [0, 1].
        # It is the job of the aircraft to scale it correctly
        raise NotImplementedError

    def shutdown(self):
        pass
