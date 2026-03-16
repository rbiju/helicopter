import queue
import numpy as np

from helicopter.utils import KeyListener, KeyConsumer, SymaCommandFactory
from .base import FlightController


class KeyboardRemote(KeyConsumer):
    def __init__(self, listener: KeyListener, channel: int = 0):
        super().__init__(listener)

        self.channel = channel
        self.thrust = 0.0
        self.pitch = 0.0
        self.yaw = 0.0
        self.trim = 0.0

        self.quit = False

    @staticmethod
    def clip(value: float):
        return min(max(value, 1.0), -1.0)

    def reset(self):
        self.thrust = 0.0
        self.pitch = 0.0
        self.yaw = 0.0

    def process(self):
        try:
            while True:
                key = self._listener.queue.get_nowait()
                if key == 'w':
                    self.thrust = self.clip(self.thrust + 0.05)
                elif key == 's':
                    self.thrust = self.clip(self.thrust - 0.05)
                elif key == 'up':
                    self.pitch = self.clip(self.pitch - 0.05)
                elif key == 'down':
                    self.pitch = self.clip(self.pitch + 0.05)
                elif key == 'right':
                    self.yaw = self.clip(self.yaw - 0.05)
                elif key == 'left':
                    self.yaw = self.clip(self.yaw + 0.05)
                elif key == 'd':
                    self.trim = self.clip(self.trim - 0.05)
                elif key == 'a':
                    self.trim = self.clip(self.trim + 0.05)
                elif key == 'q':
                    self.quit = True
                elif key == 'r':
                    self.reset()
                print(f"c:{self.channel} - t:{self.thrust} - p:{self.pitch} - y:{self.yaw} - tr:{self.trim}")
        except queue.Empty:
            pass


class ManualFlightController(FlightController):
    def __init__(self, command_factory: SymaCommandFactory = SymaCommandFactory()):
        super().__init__()
        listener = KeyListener()
        self.remote = KeyboardRemote(listener=listener)
        self.command_factory = command_factory

    def control(self):
        return np.array([self.remote.thrust, self.remote.pitch, self.remote.yaw])

    def format_command(self, command, trim=0, channel=0):
        return self.command_factory.command(thrust=command[0],
                                            pitch=command[1],
                                            yaw=command[2],
                                            trim=trim,
                                            channel=channel).format()
