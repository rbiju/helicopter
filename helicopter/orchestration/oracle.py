import numpy as np
from scipy.spatial.transform import Rotation

from helicopter.aircraft import FlightStates
from .flightplan import FlightPlan


class Oracle:
    """
    Glues together different flight plans, tracks flying state, and communicates with the orchestrator on flight status
    """
    def __init__(self, flight_plan_sequence: list[FlightPlan], tick_radius: float = 0.01):
        self.flight_plan_sequence = flight_plan_sequence
        self.tick_radius = tick_radius

        self.active_idx = 0
        self.finished = False

    @property
    def active_flight_plan(self) -> FlightPlan:
        return self.flight_plan_sequence[self.active_idx]

    @property
    def active_flight_state(self) -> FlightStates:
        if not self.finished:
            return self.active_flight_plan.flight_state
        else:
            return FlightStates.DONE

    def tick(self, r: Rotation, t: np.ndarray, timestamp: float):
        self.active_idx += 1

        if self.active_idx >= len(self.flight_plan_sequence):
            self.finished = True

        self.active_flight_plan.activate(r, t, timestamp)

    def update(self, r: Rotation, t: np.ndarray, b: np.ndarray, timestamp: float):
        if b < 0.10:
            print("Battery depleted to 10%, recharge")
            self.finished = True

        depleted = self.active_flight_plan.tick(quaternion=r, translation=t, timestamp=timestamp)
        if depleted:
            self.tick(r, t, timestamp)
