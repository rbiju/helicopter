from enum import Enum
from multiprocessing.managers import SyncManager
from threading import Lock

import numpy as np
from scipy.spatial.transform import Rotation


IDX_Q = slice(0, 4)
IDX_P = slice(4, 7)
IDX_O = slice(7, 10)
IDX_V = slice(10, 13)
IDX_BATTERY = slice(13, 14)


class FlightStates(Enum):
    IDLE = 0
    TAKEOFF = 1
    WAYPOINT_FOLLOW = 2
    HOVER = 3
    LANDING = 4
    MANUAL = 5
    DONE = 6
    KILL_POWER = 7


class Aircraft:
    def __init__(self):
        self.quaternion: Rotation = Rotation.from_quat(np.array([0.0, 0.0, 0.0, 1.0]))
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.angular_velocity = np.array([0.0, 0.0, 0.0])
        self.battery = np.array([1.0])

        self.flight_state = FlightStates.IDLE

        self._lock = Lock()

    def get_state_vector(self):
        with self._lock:
            return np.concatenate([self.quaternion.as_quat(canonical=True), self.position, self.angular_velocity, self.velocity, self.battery])

    def set_state_vector(self, state_vector: np.ndarray):
        with self._lock:
            self.quaternion = Rotation.from_quat(state_vector[IDX_Q])
            self.position = state_vector[IDX_P]
            self.angular_velocity = state_vector[IDX_O]
            self.velocity = state_vector[IDX_P]
            self.battery = state_vector[IDX_BATTERY]

    def set_flight_state(self, state: FlightStates):
        with self._lock:
            self.flight_state = state

    def set_quaternion(self, quaternion: Rotation):
        with self._lock:
            self.quaternion = quaternion

    def set_position(self, position: np.ndarray):
        with self._lock:
            self.position = position


class AircraftManager(SyncManager):
    pass

AircraftManager.register('Aircraft', Aircraft)
