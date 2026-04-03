from enum import Enum
import threading
from multiprocessing.synchronize import Lock as ProcessLock

import numpy as np
from scipy.spatial.transform import Rotation


IDX_Q = slice(0, 4)
IDX_P = slice(4, 7)
IDX_O = slice(7, 10)
IDX_V = slice(10, 13)
IDX_BATTERY = slice(13, 14)
IDX_TRIM = slice(14, 15)
IDX_STATE = slice(15, 16)
IDX_TIME = slice(16, 17)


class FlightState(Enum):
    IDLE = 0
    TAKEOFF = 1
    WAYPOINT_FOLLOW = 2
    HOVER = 3
    LANDING = 4
    MANUAL = 5
    DONE = 6
    KILL_POWER = 7

    @property
    def color(self) -> str:
        _colors = {
            self.IDLE.value: "#0A210F",
            self.TAKEOFF.value: "#14591D",
            self.WAYPOINT_FOLLOW.value: "#99AA38",
            self.HOVER.value: "#E1E289",
            self.LANDING.value: "#FFA69E",
            self.MANUAL.value: "#4F3130",
            self.DONE.value: "#ACD2ED",
            self.KILL_POWER.value: "#FF4E00"
        }
        return _colors.get(self.value, "#ffffff")


class Aircraft:
    N: int = 16
    STATE_N: int = 15
    dtype = np.float64

    def __init__(self, buffer: np.ndarray = None, lock: ProcessLock = None):
        if buffer is not None:
            self._buffer = buffer
        else:
            self._buffer = np.zeros(self.N, dtype=self.dtype)
            self._buffer[3] = 1.0
            self._buffer[13] = 1.0

        self._lock = lock if lock is not None else threading.Lock()

    @property
    def quaternion(self) -> Rotation:
        with self._lock:
            q_data = self._buffer[IDX_Q].copy()
        return Rotation.from_quat(q_data)

    @quaternion.setter
    def quaternion(self, value: Rotation):
        q_data: np.ndarray = value.as_quat(canonical=True)
        with self._lock:
            self._buffer[IDX_Q] = q_data

    @property
    def position(self) -> np.ndarray:
        with self._lock:
            return self._buffer[IDX_P].copy()

    @position.setter
    def position(self, value: np.ndarray):
        with self._lock:
            self._buffer[IDX_P] = value

    @property
    def angular_velocity(self) -> np.ndarray:
        with self._lock:
            return self._buffer[IDX_O].copy()

    @angular_velocity.setter
    def angular_velocity(self, value: np.ndarray):
        with self._lock:
            self._buffer[IDX_O] = value

    @property
    def velocity(self) -> np.ndarray:
        with self._lock:
            return self._buffer[IDX_V].copy()

    @velocity.setter
    def velocity(self, value: np.ndarray):
        with self._lock:
            self._buffer[IDX_V] = value

    @property
    def battery(self) -> float:
        with self._lock:
            return float(self._buffer[IDX_BATTERY][0])

    @battery.setter
    def battery(self, value: float):
        with self._lock:
            self._buffer[IDX_BATTERY] = np.array([value], dtype=self.dtype)

    @property
    def trim(self) -> float:
        with self._lock:
            return float(self._buffer[IDX_TRIM][0])

    @trim.setter
    def trim(self, value: float):
        with self._lock:
            self._buffer[IDX_TRIM] = np.array([value], dtype=self.dtype)

    @property
    def flight_state(self) -> FlightState:
        with self._lock:
            state_val = int(self._buffer[IDX_STATE][0])
        return FlightState(state_val)

    @flight_state.setter
    def flight_state(self, state: FlightState):
        with self._lock:
            self._buffer[IDX_STATE] = np.array([float(state.value)], dtype=self.dtype)

    @property
    def timestamp(self) -> float:
        with self._lock:
            return float(self._buffer[IDX_TIME][0])

    @timestamp.setter
    def timestamp(self, time: float):
        with self._lock:
            self._buffer[IDX_TIME] = np.array([time], dtype=self.dtype)

    def get_state_vector(self) -> np.ndarray:
        with self._lock:
            return self._buffer.copy()[:self.STATE_N]

    def set_state_vector(self, state_vector: np.ndarray):
        if state_vector.shape != (self.STATE_N,):
            raise ValueError(f"Provided vector of shape {state_vector.shape} does not match size of buffer: {self.STATE_N}")
        with self._lock:
            np.copyto(self._buffer[:self.STATE_N], state_vector)

    @classmethod
    def from_shared_memory_buffer(cls, buffer: np.ndarray, lock: ProcessLock):
        return cls(buffer=buffer, lock=lock)

    @property
    def state_dict(self) -> dict[str, np.ndarray]:
        with self._lock:
            return {'Position': self.position,
                    'Velocity': self.velocity,
                    'Orientation': Rotation.as_quat(self.quaternion, canonical=True),
                    'Angular Velocity': self.angular_velocity,
                    'Timestamp': self.timestamp,
                    'Flight State': self.flight_state,}
