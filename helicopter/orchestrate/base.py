from abc import ABC, abstractmethod
from collections import deque

import numpy as np


class FlightPlan(ABC):
    def __init__(self, dt: float, maxlen: int = 4):
        self.dt = dt
        self.maxlen = maxlen

        self._positions = self.init_positions()
        self.positions = None

        self._time = 0.

    def init_positions(self) -> deque:
        return deque(self.maxlen * [np.array([0., 0., 0.])], maxlen=self.maxlen)

    @property
    def time(self):
        return self._time

    def stack_positions(self):
        self.positions = np.stack(self._positions)

    def tick(self):
        self._time += self.dt
        self._positions.append(self.position)
        self.stack_positions()

    @property
    @abstractmethod
    def position(self) -> np.ndarray:
        raise NotImplementedError

    @property
    def velocity(self):
        return np.diff(self.positions, n=1, axis=0).mean(axis=0) / self.dt

    @property
    def acceleration(self):
        return (np.diff(self.positions, n=2, axis=0)).mean(axis=0) / (self.dt ** 2)
