from helicopter.aircraft import Aircraft
from helicopter.controller import FlightController
from helicopter.utils import Profiler, SymaRemoteControl
from helicopter.vision import ErrorStateSquareRootUnscentedKalmanFilter, D435i
from helicopter.flight_states import Done

from .oracle import Oracle


class FlightConductor:
    def __init__(self, aircraft: Aircraft,
                 controller: FlightController,
                 oracle: Oracle,
                 remote: "SymaRemoteControl",
                 ukf: ErrorStateSquareRootUnscentedKalmanFilter,
                 camera: D435i,
                 profiler: Profiler) -> None:
        """

        Args:
            aircraft: State handler, owns the integrated aircraft state, and the physical model?
            controller: The Brain
            oracle: Glues together and manages flight plans
            remote: Owns communication with the aircraft
            ukf: Fuses control inputs and vision for state estimation
            profiler:
        """
        self.aircraft = aircraft
        self.controller = controller
        self.oracle = oracle
        self.remote = remote
        self.ukf = ukf

        self.camera = camera

        self.profiler = profiler

    def homing_sequence(self):
        """
        1. Scan for the aruco markers, populate the visualizer
        2. Initialize the camera position with the triangle point matcher, use averaged points.
            Might need to randomly jitter the rotor to make the markers visible.
        """
        pass

    def loop(self):
        """
        While flight path is incomplete:

        1. Compute error between waypoint and current state, frequently check if next waypoint can be ticked
        2. Calculate control input
        3. Apply input, simulate effect and feed to Kalman filter
        4. Use predicted location as starting point for ICP, feed visual update to KF
        5. Tick to next waypoint based on the time

        while aircraft.state != FlightEnded:
            r, t, battery = aircraft.get_current_orientation()
            self.oracle.update(r, t, battery)

            flightplan = self.oracle.get_active_flightplan()
            waypoint = flightplan.waypoint

            // Could either make the waypoint mode (static, follow) live in the flightplan or in the controller
            control_signal = self.controller.get_control_signal(flightplan.waypoint, r, t)

        """
        while not isinstance(self.aircraft.flight_state, Done):
            break

