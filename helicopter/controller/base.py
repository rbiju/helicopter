from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.transform import Rotation

class FlightController(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def control(self, timestamp: float, waypoint: np.ndarray, r: Rotation, t: np.ndarray) -> float:
        # Output signals should always be on [-1, 1] or [0, 1].
        # It is the job of the aircraft to scale it correctly
        raise NotImplementedError
