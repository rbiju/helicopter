from abc import ABC
from collections import deque
import numpy as np


# TODO: add orientation getters here
class Aircraft(ABC):
    def __init__(self, mass: float, dt: float = 1 / 60, max_len: int = 4):
        self.mass = mass
        self.dt = dt

        self.max_len = max_len
        self._positions = deque(self.max_len * [np.array([0., 0., 0.])], maxlen=self.max_len)
        self.positions = None

        self.stack_positions()

    def stack_positions(self):
        self.positions = np.stack(self._positions)

    def update(self, position: np.ndarray):
        self._positions.append(position)
        self.stack_positions()

    @property
    def position(self):
        return self._positions[-1]

    @property
    def velocity(self):
        return np.diff(self.positions, n=1, axis=0).mean(axis=0) / self.dt

    @property
    def acceleration(self):
        return (np.diff(self.positions, n=2, axis=0)).mean(axis=0) / (self.dt ** 2)
