from functools import partial
from queue import Queue
from threading import Event, Lock, Thread
import traceback
import time

import numpy as np

from helicopter.aircraft import Aircraft
from helicopter.configuration import LocalHydraConfiguration
from helicopter.vision.tracking import Tracker
from helicopter.visualize import FlightVisualizer


class FakeSharedMemory:
    def __init__(self, size: int):
        self.buf = bytearray(size)

    def close(self):
        pass

    def unlink(self):
        pass

def thread_runner(target_func, kwargs_dict):
    try:
        target_func(**kwargs_dict)
    except Exception as e:
        print(f"\n[!] Exception in thread running {target_func.__name__}:")
        traceback.print_exc()


if __name__ == "__main__":
    aircraft_dummy = np.zeros(shape=(Aircraft.N,), dtype=np.float64)
    aircraft_sm = FakeSharedMemory(size=aircraft_dummy.nbytes)
    kill_event = Event()

    config = LocalHydraConfiguration('/home/ray/projects/helicopter/configs/atomic/tracker.yaml')
    tracker_partial: partial = Tracker.from_hydra_configuration(config)
    tracker: Tracker = tracker_partial(aircraft_sm=aircraft_sm, kill_signal=kill_event)

    vis_config = LocalHydraConfiguration('/home/ray/projects/helicopter/configs/atomic/flight_visualizer.yaml')
    visualizer_partial: partial = FlightVisualizer.from_hydra_configuration(vis_config)
    visualizer: FlightVisualizer = visualizer_partial(aircraft_sm=aircraft_sm, kill_signal=kill_event)

    marker_queue = Queue()
    origin_queue = Queue()
    lock = Lock()

    tracker_thread = Thread(
        target=thread_runner,
        args=(tracker.initialize, {
            'marker_queue': marker_queue,
            'origin_queue': origin_queue,
            'aircraft_lock': lock
        }),
        daemon=True
    )

    visualizer_thread = Thread(
        target=thread_runner,
        args=(visualizer.initialize, {
            'marker_queue': marker_queue,
            'origin_queue': origin_queue,
            'aircraft_lock': lock
        }),
        daemon=True
    )

    try:
        tracker_thread.start()
        visualizer_thread.start()

        while tracker_thread.is_alive() and visualizer_thread.is_alive():
            time.sleep(0.1)

        if tracker_thread.is_alive() or visualizer_thread.is_alive():
            print("\n[!] Error: A thread died. Exiting to kill the stuck thread.")
            kill_event.set()
        else:
            print('Both initialized successfully')

    finally:
        tracker.cleanup()
        visualizer.cleanup()