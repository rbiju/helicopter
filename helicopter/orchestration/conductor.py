import time

from helicopter.aircraft import Aircraft
from helicopter.controller import FlightController
from helicopter.utils import Profiler, SymaRemoteControl

from .oracle import Oracle
from .flightplan import TakeOffFlightPlan, HoverFlightPlan


class FlightConductor:
    def __init__(self, aircraft: Aircraft,
                 controller: FlightController,
                 oracle: Oracle,
                 remote: SymaRemoteControl,
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

        self.hover_oracle = Oracle(flight_plan_sequence=[TakeOffFlightPlan(takeoff_height=0.2),
                                                         HoverFlightPlan(hover_time=10.0)])

    def homing_sequence(self):
        """
        1. Initialize the helicopter position with the triangle point matcher, use averaged points.
            Might need to randomly jitter the rotor to make the markers visible.
        2. Small hover to adjust trim
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
        flight_start_time = time.time()
        while not self.aircraft.flight_state == 5:
            timestamp = time.time() - flight_start_time
            r, t, battery = self.aircraft.quaternion, self.aircraft.position, self.aircraft.battery
            self.oracle.update(r, t, battery, timestamp=timestamp)

            flightplan = self.oracle.active_flight_plan
            errors = flightplan.compute_error(quaternion=r, translation=t)
            commands = self.controller.control(timestamp, errors)

            formatted_commands = self.controller.format_command(commands, 0.0, 128)
            self.remote.send_command(formatted_commands)
