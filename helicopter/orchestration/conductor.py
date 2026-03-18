import time
from multiprocessing.synchronize import Lock, Event
from multiprocessing.shared_memory import SharedMemory

import numpy as np

from helicopter.configuration import HydraConfigurable
from helicopter.aircraft import Aircraft
from helicopter.controller import FlightController
from helicopter.utils import Profiler, SymaRemoteControl

from .oracle import Oracle


@HydraConfigurable
class FlightConductor:
    def __init__(self, aircraft: Aircraft,
                 controller: FlightController,
                 oracle: Oracle,
                 remote: SymaRemoteControl,
                 kill_signal: Event) -> None:
        """

        Args:
            controller: The Brain
            oracle: Glues together and manages flight plans
            remote: Owns communication with the aircraft
        """
        self.aircraft = aircraft
        self.controller = controller
        self.oracle = oracle
        self.remote = remote

        self.profiler = Profiler()

        self.kill_signal = kill_signal

    def initialize(self):
        # TODO: load waypoints for visualization
        pass

    def loop(self, command_sm: SharedMemory, lock: Lock):
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
        """
        flight_start_time = time.time()

        while not self.oracle.finished:
            if self.kill_signal.is_set():
                raise RuntimeError('Conductor detected kill signal. Shutting down')

            timestamp = time.time() - flight_start_time
            r, t, battery = self.aircraft.quaternion, self.aircraft.position, self.aircraft.battery
            self.oracle.update(r, t, timestamp=timestamp)

            flightplan = self.oracle.active_flight_plan
            errors = flightplan.compute_error(quaternion=r, translation=t)
            commands = self.controller.control(timestamp=timestamp, errors=errors)

            if self.controller.killed:
                self.kill_signal.set()

            formatted_commands = self.controller.format_command(commands, 0.0, 128)
            command_sent = self.remote.send_command(formatted_commands)
            if command_sent:
                buffer = np.ndarray(commands.shape, dtype=commands.dtype, buffer=command_sm.buf)
                with lock:
                    np.copyto(buffer, command_sm)

    def cleanup(self):
        self.controller.shutdown()
        self.remote.shutdown()
