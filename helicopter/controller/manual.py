import numpy as np
from scipy.spatial.transform import Rotation

from helicopter.flightplan import FlightPlan
from helicopter.remote import RemoteRecorderThread
from helicopter.utils import ArduinoLoader
from .base import FlightController


class ManualFlightController(FlightController):
    def __init__(self, recorder_thread: RemoteRecorderThread,
                 arduino_loader: ArduinoLoader = ArduinoLoader(sketch_path='py_recorder')):
        super().__init__(arduino_loader=arduino_loader)

        self.recorder_thread = recorder_thread
        self.recorder_thread.start()

    def control(self, flightplan: FlightPlan,
                quaternion: Rotation,
                position: np.ndarray,
                timestamp: float) -> np.ndarray:
        commands = self.recorder_thread.get_commands()
        return commands

    def shutdown(self):
        self.recorder_thread.end()
