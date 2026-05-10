from multiprocessing.shared_memory import SharedMemory
from multiprocessing.synchronize import Lock, Event

import numpy as np

from helicopter.configuration import HydraConfigurable
from helicopter.aircraft import Aircraft
from helicopter.controller import FlightController
from helicopter.utils import Profiler, CommandBufferConstants

from .oracle import Oracle


@HydraConfigurable
class FlightConductor:
    def __init__(self, aircraft_sm: SharedMemory,
                 controller: FlightController,
                 oracle: Oracle,
                 kill_signal: Event) -> None:
        self.aircraft_buffer = np.ndarray(shape=(Aircraft.N,),
                                          dtype=Aircraft.dtype,
                                          buffer=aircraft_sm.buf)
        self.aircraft = None

        self.controller = controller
        self.oracle = oracle

        self.profiler = Profiler()

        self.kill_signal = kill_signal

    def initialize(self, aircraft_lock: Lock):
        # TODO: load waypoints for visualization
        if self.aircraft is None:
            self.aircraft = Aircraft(buffer=self.aircraft_buffer, lock=aircraft_lock)

        self.oracle.active_flight_plan.activate(
            quaternion=self.aircraft.quaternion,
            position=self.aircraft.position,
            timestamp=0.0
        )
        self.aircraft.flight_state = self.oracle.active_flight_state(timestamp=0.0)

    def loop(self, command_sm: SharedMemory, lock: Lock):
        """
        While flight path is incomplete:

        1. Compute error between waypoint and current state, frequently check if next waypoint can be ticked
        2. Calculate control input
        3. Apply input, simulate effect and feed to Kalman filter
        4. Use predicted location as starting point for ICP, feed visual update to KF
        5. Tick to next waypoint based on the time
        """
        command_buffer = np.ndarray(shape=(CommandBufferConstants.N,),
                                    dtype=CommandBufferConstants.dtype,
                                    buffer=command_sm.buf)
        while not self.oracle.finished:
            if self.kill_signal.is_set():
                print('Conductor detected kill signal. Shutting down')
                break

            timestamp = self.aircraft.timestamp
            self.aircraft.flight_state = self.oracle.active_flight_state(timestamp)
            r, t = self.aircraft.quaternion, self.aircraft.position
            self.oracle.update(r, t, timestamp=timestamp)
            if self.oracle.finished:
                break

            flightplan = self.oracle.active_flight_plan
            sent_command = self.controller.control(flightplan, r, t, timestamp)
            with lock:
                np.copyto(command_buffer, sent_command)

        print('Flight plans exhausted. Ending Flight!')
        self.kill_signal.set()

    def cleanup(self):
        self.controller.shutdown()
