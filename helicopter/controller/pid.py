from typing import NamedTuple

import numpy as np
from scipy.spatial.transform import Rotation

from helicopter.utils import Command
from .base import FlightController


class PIDGains(NamedTuple):
    k_p: float
    k_i: float
    k_d: float


class PIDController:
    def __init__(self, gains: PIDGains,
                 lambd: float = 0.9,
                 max_value: float = 1.0,
                 min_value: float = -1.0):
        super().__init__()
        self.prev_error = 0.

        self.accumulator = 0.

        self.gains = gains

        self.lambd = lambd

        self.max_value = max_value
        self.min_value = min_value

    def proportional(self, error: float) -> float:
        return self.gains.k_p * error

    def integral(self, error: float, dt) -> float:
        self.accumulator += (error * dt)
        return self.gains.k_i * self.accumulator

    def derivative(self, error: float, dt) -> float:
        return self.gains.k_d * (error - self.prev_error) / dt

    def reset(self):
        self.accumulator = 0.
        self.prev_error = 0.

    @staticmethod
    def xnor(a, b):
        return (a and b) or (not a and not b)

    def control(self, dt, error: float) -> float:
        error = self.lambd * error + (1 - self.lambd) * self.prev_error

        out = self.proportional(error) + self.integral(error, dt) + self.derivative(error, dt)

        if abs(out) > 1.0:
            out = max(min(out, self.max_value), self.min_value)
            clamped = True
        else:
            clamped = False

        if self.xnor(error <= 0, out <= 0):
            sign = True
        else:
            sign = False

        if clamped and sign:
            self.accumulator = 0.

        self.prev_error = error

        return out


class HelicopterPIDController(FlightController):
    def __init__(self, throttle: PIDController, pitch: PIDController, yaw: PIDController):
        super().__init__()
        self.throttle = throttle
        self.pitch = pitch
        self.yaw = yaw

        self.trim = PIDController(gains=PIDGains(k_p = 0.1, k_i = 0.0, k_d = 0.0))

        self.controllers = [self.throttle, self.pitch, self.yaw]

        self.last_time = 0.0

    def reset(self):
        for controller in self.controllers:
            controller.reset()

    @staticmethod
    def compute_pid_errors(waypoint: np.ndarray, r: Rotation, t: np.ndarray):
        position_error = waypoint - t
        e_throttle = position_error[2]

        yaw_rotvec = Rotation.as_rotvec(r)[2]
        yaw_rotation = Rotation.from_rotvec(np.array([0.0, 0.0, yaw_rotvec]))
        position_error_body = yaw_rotation.inv().apply(position_error)
        e_pitch = position_error_body[0]

        if e_pitch < 0.05:
            e_yaw = 0.0
        else:
            psi_target = np.arctan2(position_error[1], position_error[0])
            psi_current = yaw_rotvec
            e_yaw = psi_target - psi_current

        return np.array([e_throttle, e_pitch, e_yaw])

    def control(self, timestamp: float, errors: np.ndarray):
        commands = []
        for i in range(3):
            command = self.controllers[i].control(errors[i], timestamp - self.last_time)
            commands.append(command)

        self.last_time = timestamp

        return np.array(commands)

    @staticmethod
    def format_command(command, trim, channel=128):
        return Command(throttle=command[0],
                       pitch=command[1],
                       yaw=command[2],
                       trim=trim,
                       channel=channel).format()
