from abc import ABC, abstractmethod
from dataclasses import dataclass
import threading

import numpy as np

READY_ACK = 129


@dataclass
class ControlPacket:
    throttle: float
    pitch: float
    yaw: float


@dataclass
class RecordingPacket:
    channel: int
    yaw: float
    pitch: float
    throttle: float
    trim: int

    @classmethod
    def from_list(cls, vals: list) -> "RecordingPacket":
        return cls(*vals)


class RemoteState(ABC):
    def __init__(self):
        self.channel = 128
        self.yaw = 63
        self.pitch = 63
        self.throttle = 0
        self.trim = 0

    def as_list(self):
        return [self.channel, self.yaw, self.pitch, self.throttle, self.trim]

    def convert_to_float(self):
        throttle = self.throttle / 127.
        pitch = ((128 - (self.pitch + 1)) - 64.) / 128. * 2
        yaw = ((self.yaw + 1) - 64.) / 128. * 2
        return np.array([throttle, pitch, yaw])

    @abstractmethod
    def update(self, commands):
        raise NotImplementedError


class RemoteThread(ABC, threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True
        self.lock = threading.Lock()

    @abstractmethod
    def run(self):
        raise NotImplementedError

    @abstractmethod
    def shutdown(self):
        raise NotImplementedError

    def stop(self):
        self.running = False
        self.join()
        self.shutdown()
