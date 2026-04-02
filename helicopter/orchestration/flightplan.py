from abc import ABC, abstractmethod
from collections import deque

import numpy as np
from scipy.spatial.transform import Rotation

from helicopter.aircraft import FlightStates


class FlightPlan(ABC):
    def __init__(self):
        self._waypoints = deque()
        self.activated = False

    @abstractmethod
    def flight_state(self, timestamp: float) -> FlightStates:
        raise NotImplementedError

    @property
    def waypoint(self) ->  np.ndarray:
        return self._waypoints[0]

    @abstractmethod
    def compute_error(self, quaternion: Rotation, translation: np.ndarray):
        raise NotImplementedError

    @abstractmethod
    def activate(self, quaternion: Rotation, translation: np.ndarray, timestamp: float):
        raise NotImplementedError

    def _advance_waypoint(self):
        try:
            self._waypoints.popleft()
            return False
        except IndexError:
            return True

    @abstractmethod
    def tick(self, quaternion: Rotation, translation: np.ndarray, timestamp: float):
        raise NotImplementedError


class WaypointFollowingFlightPlan(FlightPlan, ABC):
    def __init__(self):
        super().__init__()

    def compute_error(self, r: Rotation, t: np.ndarray):
        position_error = self.waypoint - t
        e_throttle = position_error[2]

        yaw_rotvec = Rotation.as_rotvec(r)[2]
        yaw_rotation = Rotation.from_rotvec(np.array([0.0, 0.0, yaw_rotvec]))
        position_error_body = yaw_rotation.inv().apply(position_error)
        e_pitch = position_error_body[0]

        if np.abs(e_pitch) < 0.05:
            e_yaw = 0.0
        else:
            psi_target = np.arctan2(position_error[1], position_error[0])
            psi_current = yaw_rotvec
            e_yaw = psi_target - psi_current

        return np.array([e_throttle, e_pitch, e_yaw])


class ConstantHeadingFlightPlan(FlightPlan, ABC):
    def __init__(self):
        super().__init__()
        self.reference_heading = 0.0

    def compute_error(self, quaternion: Rotation, t: np.ndarray):
        position_error = self.waypoint - t
        e_throttle = position_error[2]

        yaw_rotvec = Rotation.as_rotvec(quaternion)[2]
        yaw_rotation = Rotation.from_rotvec(np.array([0.0, 0.0, yaw_rotvec]))
        position_error_body = yaw_rotation.inv().apply(position_error)
        e_pitch = position_error_body[0]

        e_yaw = self.reference_heading - yaw_rotvec

        return np.array([e_throttle, e_pitch, e_yaw])


class IdleFlightPlan(FlightPlan, ABC):
    def __init__(self):
        super().__init__()

    def flight_state(self, timestamp: float) -> FlightStates:
        return FlightStates.IDLE

    def compute_error(self, quaternion: Rotation, translation: np.ndarray):
        return np.array([0.0, 0.0, 0.0])

    def activate(self, quaternion: Rotation, translation: np.ndarray, timestamp: float):
        pass

    def tick(self, quaternion: Rotation, translation: np.ndarray, timestamp: float):
        return True


class TakeOffFlightPlan(WaypointFollowingFlightPlan):
    def __init__(self, takeoff_height: float = 0.2, ground_time: float = 0.1):
        super().__init__()
        self.takeoff_height = takeoff_height
        self.tick_radius = 0.025
        self.ground_time = ground_time
        self.start_time = 0

    def flight_state(self, timestamp: float) -> FlightStates:
        if timestamp - self.start_time < self.ground_time:
            return FlightStates.IDLE
        else:
            return FlightStates.TAKEOFF

    def activate(self, quaternion: Rotation, translation: np.ndarray, timestamp: float):
        self._waypoints.append(translation + np.array([0, 0, self.takeoff_height]))
        self.start_time = timestamp
        self.activated = True

    def tick(self, quaternion: Rotation, translation: np.ndarray, timestamp: float):
        depleted = False
        if np.linalg.norm(translation - self.waypoint) < self.tick_radius:
            depleted = self._advance_waypoint()
        return depleted


class HoverFlightPlan(ConstantHeadingFlightPlan):
    def __init__(self, hover_time: float = 10.0):
        super().__init__()
        self.start_time = 0
        self.hover_time = hover_time

    def flight_state(self, timestamp: float) -> FlightStates:
        return FlightStates.HOVER

    def activate(self, quaternion: Rotation, translation: np.ndarray, timestamp: float):
        self.reference_heading = Rotation.as_rotvec(quaternion)[2]
        self.start_time = timestamp
        self._waypoints.append(translation)
        self.activated = True

    def tick(self, quaternion: Rotation, translation: np.ndarray, timestamp: float):
        depleted = False
        if (timestamp - self.start_time) > self.hover_time:
            depleted = self._advance_waypoint()
        return depleted


class ManualFlightPlan(FlightPlan):
    def __init__(self, flight_time: float = 120.0):
        super().__init__()
        self.start_time = 0
        self.hover_time = flight_time

    def flight_state(self, timestamp: float) -> FlightStates:
        return FlightStates.MANUAL

    def compute_error(self, r: Rotation, t: np.ndarray):
        return np.array([0.0, 0.0, 0.0])

    def activate(self, quaternion: Rotation, translation: np.ndarray, timestamp: float):
        self.activated = True
        self.start_time = timestamp

    def tick(self, quaternion: Rotation, translation: np.ndarray, timestamp: float):
        depleted = False
        if (timestamp - self.start_time) > self.hover_time:
            depleted = True
        return depleted
