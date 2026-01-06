from typing import NamedTuple

from .base import FlightController, Aircraft


class PIDGains(NamedTuple):
    k_p: float
    k_i: float
    k_d: float


class PIDController(FlightController):
    def __init__(self, craft: Aircraft,
                 gains: PIDGains,
                 pv: str,
                 lambd: float = 0.9,):
        super().__init__(craft=craft)
        if not hasattr(craft, pv):
            raise AttributeError(f"Craft {craft} has no attribute {pv}")
        self.pv = pv

        self.prev_error = 0.

        self.accumulator = 0.

        self.gains = gains

        self.lambd = lambd

    def proportional(self, error: float) -> float:
        return self.gains.k_p * error

    def integral(self, error: float, dt) -> float:
        self.accumulator += (error * dt)
        return self.gains.k_i * self.accumulator

    def derivative(self, error: float, dt) -> float:
        return self.gains.k_d * (error - self.prev_error) / dt

    @staticmethod
    def xnor(a, b):
        return (a and b) or (not a and not b)

    def control(self, dt) -> float:
        measurement: float = getattr(self.craft, self.pv)
        error = self.setpoint - measurement

        error = self.lambd * error + (1 - self.lambd) * self.prev_error

        out = self.proportional(error) + self.integral(error, dt) + self.derivative(error, dt)

        if abs(out) > 1.0:
            out = max(min(out, 1.0), -1.0)
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
