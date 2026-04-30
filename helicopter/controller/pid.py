from typing import NamedTuple

import numpy as np
from scipy.spatial.transform import Rotation

from helicopter.flightplan import FlightPlan
from helicopter.remote import RemoteControlThread, ControlPacket
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


class PIDFlightController(FlightController):
    def __init__(self, throttle: PIDController, pitch: PIDController, yaw: PIDController,
                 remote_thread: RemoteControlThread):
        super().__init__()
        self.throttle = throttle
        self.pitch = pitch
        self.yaw = yaw

        self.remote_thread = remote_thread

        # This order determines the order of the command array
        self.controllers = [self.throttle, self.pitch, self.yaw]

        self.last_time = 0.0
        self.remote_thread.start()

    def reset(self):
        for controller in self.controllers:
            controller.reset()

    def get_command(self, timestamp: float, errors: np.ndarray) -> ControlPacket:
        commands = []
        for i in range(len(self.controllers)):
            command = self.controllers[i].control(errors[i], timestamp - self.last_time)
            commands.append(command)

        self.last_time = timestamp

        return ControlPacket(*commands)

    def control(self, flightplan: FlightPlan,
                quaternion: Rotation,
                position: np.ndarray,
                timestamp: float) -> np.ndarray:
        error = flightplan.compute_error(quaternion=quaternion, position=position)
        commands = self.get_command(timestamp, error)
        self.remote_thread.update(commands)
        self.last_time = timestamp

        return self.remote_thread.most_recently_sent()

    def shutdown(self):
        self.remote_thread.stop()
