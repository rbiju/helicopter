from abc import ABC, abstractmethod
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation


class FlightPlan(ABC):
    def __init__(self):
        self._waypoints = deque()
        self.activated = False

    @property
    def waypoint(self) ->  np.ndarray:
        return self._waypoints[0]

    @abstractmethod
    def activate(self, quaternion: Rotation, translation: np.ndarray, time: float):
        raise NotImplementedError

    def tick(self):
        try:
            self._waypoints.popleft()
            return False
        except IndexError:
            return True


class TakeOffFlightPlan(FlightPlan):
    def __init__(self, takeoff_height: float = 0.2):
        super().__init__()
        self.takeoff_height = takeoff_height

    def activate(self, quaternion: Rotation, translation: np.ndarray, time: float):
        self._waypoints.append(translation + np.array([0, 0, self.takeoff_height]))
        self.activated = True
