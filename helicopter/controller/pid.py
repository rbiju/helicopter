from typing import NamedTuple

import numpy as np
from scipy.spatial.transform import Rotation

from helicopter.flightplan import FlightPlan
from helicopter.remote import RemoteControlThread, ControlPacket
from helicopter.utils import ArduinoLoader

from .base import FlightController
from .logger import PIDLogger


class PIDGains(NamedTuple):
    k_p: float
    k_i: float
    k_d: float
    feedforward: float = 0.0


class PIDController:
    def __init__(self, gains: PIDGains,
                 max_value: float = 1.0,
                 min_value: float = -1.0,
                 lambd: float = 0.1,
                 ):
        super().__init__()
        self.gains = gains

        self.max_value = max_value
        self.min_value = min_value
        self.lambd = lambd

        self.proportional_value = 0.0
        self.integral_value = 0.0
        self.derivative_value = 0.0

        self.prev_error = 0.
        self.prev_derivative = 0.
        self.accumulator = 0.

    def proportional(self, error: float) -> float:
        self.proportional_value = self.gains.k_p * error
        return self.proportional_value

    def integral(self, error: float, dt) -> float:
        self.accumulator += (error * dt)
        self.integral_value = self.gains.k_i * self.accumulator
        return self.integral_value

    def derivative(self, error: float, dt: float) -> float:
        raw_derivative = (error - self.prev_error) / dt
        filtered_derivative = (self.lambd * raw_derivative) + ((1 - self.lambd) * self.prev_derivative)

        self.prev_derivative = filtered_derivative
        self.derivative_value = self.gains.k_d * filtered_derivative

        return self.derivative_value

    def reset(self):
        self.accumulator = 0.
        self.prev_derivative = 0.

    @property
    def state(self) -> np.ndarray:
        return np.array([self.proportional_value,
                         self.integral_value,
                         self.derivative_value,
                         self.accumulator])

    @staticmethod
    def xnor(a, b):
        return (a and b) or (not a and not b)

    def control(self, error: float, dt: float) -> float:
        out = (self.proportional(error) +
               self.integral(error, dt) +
               self.derivative(error, dt) +
               self.gains.feedforward)

        if out > self.max_value or out < self.min_value:
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
                 remote_thread: RemoteControlThread,
                 arduino_loader: ArduinoLoader = ArduinoLoader(sketch_path='py_controller')):
        super().__init__(arduino_loader=arduino_loader)
        self.throttle = throttle
        self.pitch = pitch
        self.yaw = yaw
        self.remote_thread = remote_thread

        self.logger = PIDLogger()

        # This order determines the order of the command array
        self._controllers = [self.throttle, self.pitch, self.yaw]

        self.last_time = 0.0
        self.remote_thread.start()

    def reset(self):
        for controller in self._controllers:
            controller.reset()

    def get_command(self, timestamp: float, errors: np.ndarray) -> ControlPacket | None:
        dt = timestamp - self.last_time

        if dt > 1e-5:
            commands = []
            for i in range(len(self._controllers)):
                command = self._controllers[i].control(dt=dt, error=errors[i])
                commands.append(command)

            self.last_time = timestamp
            return ControlPacket(*commands)
        else:
            return None

    def control(self, flightplan: FlightPlan,
                quaternion: Rotation,
                position: np.ndarray,
                timestamp: float) -> np.ndarray:
        if timestamp != self.last_time:
            error = flightplan.compute_error(quaternion=quaternion, position=position)
            commands = self.get_command(timestamp=timestamp, errors=error)
            self.logger.log(timestamp=timestamp,
                            error=error,
                            controller_states=[controller.state for controller in self._controllers])
            self.remote_thread.update(commands)
            self.last_time = timestamp

        return self.remote_thread.most_recently_sent()

    def shutdown(self):
        self.remote_thread.end()
        self.logger.save()
