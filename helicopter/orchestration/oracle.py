import numpy as np
from scipy.spatial.transform import Rotation

from helicopter.flight_states import flight_state_registry, FlightState
from .flightplan import FlightPlan


class Oracle:
    """
    Glues together different flight plans, tracks flying state, and communicates with the orchestrator on flight status
    """
    def __init__(self, flight_plan_sequence: list[dict[str, FlightPlan]], tick_radius: float = 0.01):
        self.flight_plan_sequence = flight_plan_sequence
        self.tick_radius = tick_radius

        self.active_idx = 0
        self.active_flight_plan = self.get_active_flight_plan(self.active_idx)
        self.active_flight_state = flight_state_registry[list(self.flight_plan_sequence[self.active_idx].keys())[0]]

        self.finished = False

    def get_active_flight_plan(self, idx) -> FlightPlan:
        return list(self.flight_plan_sequence[idx].values())[0]

    def get_active_flight_state(self, idx) -> FlightState:
        return flight_state_registry[list(self.flight_plan_sequence[idx].keys())[0]]

    def tick(self, r: Rotation, t: np.ndarray, timestamp: float):
        self.active_idx += 1

        if self.active_idx >= len(self.flight_plan_sequence):
            self.finished = True

        self.active_flight_plan = self.get_active_flight_plan(self.active_idx)
        self.active_flight_state = self.get_active_flight_state(self.active_idx)

        self.active_flight_plan.activate(r, t, timestamp)

    def update(self, r: Rotation, t: np.ndarray, b: np.ndarray, timestamp: float):
        if b < 0.10:
            print("Battery depleted to 10%, recharge")
            self.finished = True

        if np.linalg.norm(t - self.active_flight_plan.waypoint) < self.tick_radius:
            depleted = self.active_flight_plan.tick()
            if depleted:
                self.tick(r, t, timestamp)
