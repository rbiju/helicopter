from functools import partial
from threading import Event, Lock

import numpy as np

from helicopter.aircraft import Aircraft
from helicopter.configuration import LocalHydraConfiguration
from helicopter.vision.tracking import Tracker


class FakeSharedMemory:
    def __init__(self, size: int):
        self.buf = bytearray(size)
    def close(self):
        pass
    def unlink(self):
        pass


if __name__ == "__main__":
    aircraft_dummy = np.zeros(shape=(Aircraft.N,), dtype=np.float64)
    aircraft_sm = FakeSharedMemory(size=aircraft_dummy.nbytes)
    kill_event = Event()

    config = LocalHydraConfiguration('/home/ray/projects/helicopter/configs/atomic/tracker.yaml')
    tracker: partial = Tracker.from_hydra_configuration(config)
    tracker: Tracker = tracker(aircraft_sm=aircraft_sm,
                               kill_signal=kill_event,)

    ready_event = Event()
    image_dummy = np.zeros((720, 1280), dtype=np.uint8)
    image_sm = FakeSharedMemory(size=image_dummy.nbytes)
    quat_dummy = np.zeros(4, dtype=np.float64)
    quat_sm = FakeSharedMemory(size=quat_dummy.nbytes)
    lock = Lock()
    intrinsics = {}

    try:
        tracker.initialize(quat_sm=quat_sm,
                           image_sm=image_sm,
                           intrinsics=intrinsics,
                           aircraft_lock=lock,)
        print('initialized')
    finally:
        tracker.cleanup()
