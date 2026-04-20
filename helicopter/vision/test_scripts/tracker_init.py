from functools import partial
from queue import Queue
from threading import Event, Lock

import numpy as np
from scipy.spatial.transform import Rotation

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

    marker_queue = Queue()
    origin_queue = Queue()
    orientation_ready = Event()
    origin_queue.put([{'id': 0,
                      'position': np.array([-0.355 + 0.035, -0.685 + 0.035, 0]),
                      'rotation': Rotation.from_euler('y', [90], degrees=True)},
                      {'id': 1,
                       'position': np.array([-0.355, -0.685 - 0.035, -0.035]),
                       'rotation': Rotation.from_euler('z', [90], degrees=True)},
                      {'id': 2,
                       'position': np.array([-0.355, -0.685, -0.035]),
                       'rotation': Rotation.from_euler('z', [0], degrees=True)},
                      ])
    lock = Lock()

    try:
        tracker.initialize(marker_queue=marker_queue,
                           origin_queue=origin_queue,
                           orientation_ready=orientation_ready,
                           aircraft_lock=lock,)
        print('initialized')
    finally:
        tracker.cleanup()
