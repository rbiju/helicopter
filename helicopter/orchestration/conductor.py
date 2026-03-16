import time
from multiprocessing.synchronize import Lock

from helicopter.configuration import HydraConfigurable
from helicopter.aircraft import Aircraft
from helicopter.controller import FlightController
from helicopter.utils import Profiler, SymaRemoteControl

from .oracle import Oracle


@HydraConfigurable
class FlightConductor:
    def __init__(self, aircraft: Aircraft,
                 aircraft_lock: Lock,
                 controller: FlightController,
                 oracle: Oracle,
                 remote: SymaRemoteControl,
                 profiler: Profiler) -> None:
        """

        Args:
            controller: The Brain
            oracle: Glues together and manages flight plans
            remote: Owns communication with the aircraft
            profiler:
        """
        self.aircraft = aircraft
        self.aircraft_lock = aircraft_lock
        self.controller = controller
        self.oracle = oracle
        self.remote = remote

        self.profiler = profiler

    def initialize(self):
        # TODO: load waypoints for visualization
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
        flight_start_time = time.time()

        while not self.oracle.finished:
            with self.aircraft_lock:
                flight_state = self.aircraft.flight_state

            timestamp = time.time() - flight_start_time
            r, t, battery = self.aircraft.quaternion, self.aircraft.position, self.aircraft.battery
            self.oracle.update(r, t, battery, timestamp=timestamp)

            flightplan = self.oracle.active_flight_plan
            errors = flightplan.compute_error(quaternion=r, translation=t)
            commands = self.controller.control(timestamp=timestamp, errors=errors)

            formatted_commands = self.controller.format_command(commands, 0.0, 128)
            self.remote.send_command(formatted_commands)
