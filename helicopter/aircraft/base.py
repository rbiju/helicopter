from enum import Enum
from multiprocessing.managers import SyncManager
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Lock

import numpy as np
from scipy.spatial.transform import Rotation


IDX_Q = slice(0, 4)
IDX_P = slice(4, 7)
IDX_O = slice(7, 10)
IDX_V = slice(10, 13)
IDX_BATTERY = slice(13, 14)
IDX_TRIM = slice(14, 15)


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
    N: int = 16
    dtype = np.float64

    def __init__(self):
        self.quaternion: Rotation = Rotation.from_quat(np.array([0.0, 0.0, 0.0, 1.0]))
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.angular_velocity = np.array([0.0, 0.0, 0.0])
        self.battery = np.array([1.0])
        self.trim = np.array([0.0])

    def get_state_vector(self):
        return np.concatenate([self.quaternion.as_quat(canonical=True),
                               self.position,
                               self.angular_velocity,
                               self.velocity,
                               self.battery,
                               self.trim])

    def set_state_vector(self, state_vector: np.ndarray):
        self.quaternion = Rotation.from_quat(state_vector[IDX_Q])
        self.position = state_vector[IDX_P]
        self.angular_velocity = state_vector[IDX_O]
        self.velocity = state_vector[IDX_P]
        self.battery = state_vector[IDX_BATTERY]
        self.trim = state_vector[IDX_TRIM]

    @classmethod
    def from_state_vector(cls, state_vector: np.ndarray):
        cls.quaternion = Rotation.from_quat(state_vector[IDX_Q])
        cls.position = state_vector[IDX_P]
        cls.angular_velocity = state_vector[IDX_O]
        cls.velocity = state_vector[IDX_P]
        cls.battery = state_vector[IDX_BATTERY]
        cls.trim = state_vector[IDX_TRIM]

        return cls

    @classmethod
    def from_shared_memory_buffer(cls, buffer: np.ndarray, lock: Lock):
        shape = (cls.N,)
        local_state = np.empty(shape, dtype=np.float64)
        with lock:
            np.copyto(local_state, buffer)

        return cls.from_state_vector(local_state)

    def set_quaternion(self, quaternion: Rotation):
        self.quaternion = quaternion

    def set_position(self, position: np.ndarray):
        self.position = position


class AircraftManager(SyncManager):
    pass

AircraftManager.register('Aircraft', Aircraft)
