import numpy as np
from scipy.spatial.transform import Rotation

from helicopter.aircraft import FlightState
from .flightplan import FlightPlan


class Oracle:
    """
    Glues together different flight plans,
    tracks flying state,
    and communicates with the orchestrator on flight status
    """
    def __init__(self, flight_plan_sequence: list[FlightPlan]):
        self.flight_plan_sequence = flight_plan_sequence

        self.active_idx = 0
        self.finished = False

    @property
    def active_flight_plan(self) -> FlightPlan:
        return self.flight_plan_sequence[self.active_idx]

    def active_flight_state(self, timestamp: float) -> FlightState:
        if not self.finished:
            return self.active_flight_plan.flight_state(timestamp=timestamp)
        else:
            return FlightState.DONE

    def kill_flight(self):
        self.finished = True

    def tick(self, r: Rotation, t: np.ndarray, timestamp: float):
        self.active_idx += 1

        if self.active_idx >= len(self.flight_plan_sequence):
            self.finished = True
            return

        self.active_flight_plan.activate(r, t, timestamp)

    def update(self, r: Rotation, t: np.ndarray, timestamp: float):
        depleted = self.active_flight_plan.tick(quaternion=r, translation=t, timestamp=timestamp)
        if depleted:
            self.tick(r, t, timestamp)

    def add_flight_plan(self, flight_plan: FlightPlan):
        self.flight_plan_sequence.append(flight_plan)
