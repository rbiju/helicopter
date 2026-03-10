from multiprocessing.managers import BaseManager

import numpy as np
from scipy.spatial.transform import Rotation

from helicopter.flight_states import flight_state_registry

IDX_Q = slice(0, 4)
IDX_P = slice(4, 7)
IDX_O = slice(7, 10)
IDX_V = slice(10, 13)
IDX_BATTERY = slice(13, 14)


class Aircraft:
    def __init__(self):
        self.quaternion = Rotation.from_quat(np.array([0.0, 0.0, 0.0, 1.0]))
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.angular_velocity = np.array([0.0, 0.0, 0.0])
        self.battery = np.array([1.0])

        self.flight_state = flight_state_registry.get_class('Idle')()

    def get_state_vector(self):
        return np.concatenate([self.quaternion.as_quat(canonical=True), self.position, self.angular_velocity, self.velocity, self.battery])

    def set_state_vector(self, state_vector: np.ndarray):
        self.quaternion = Rotation.from_quat(state_vector[IDX_Q])
        self.position = state_vector[IDX_P]
        self.angular_velocity = state_vector[IDX_O]
        self.velocity = state_vector[IDX_P]
        self.battery = state_vector[IDX_BATTERY]

    def set_flight_state(self, state: str):
        self.flight_state = flight_state_registry.get_class(state)()

    def set_quaternion(self, quaternion: Rotation):
        self.quaternion = Rotation.as_quat(quaternion, canonical=True)

    def set_position(self, position: np.ndarray):
        self.position = position


class AircraftManager(BaseManager):
    pass

AircraftManager.register('Aircraft', Aircraft)
