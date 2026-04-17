from functools import partial
from queue import Queue
from threading import Event, Lock, Thread
import traceback
import time

import numpy as np
from scipy.spatial.transform import Rotation

from helicopter.aircraft import Aircraft
from helicopter.configuration import LocalHydraConfiguration
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
    lock = Lock()

    mock_buffer = np.ndarray(shape=(Aircraft.N,), dtype=np.float64, buffer=aircraft_sm.buf)
    mock_aircraft = Aircraft(buffer=mock_buffer, lock=lock)
    mock_aircraft.position = np.array([0.0085546, -0.42317, 0])
    mock_aircraft.quaternion = Rotation.identity()

    kill_event = Event()

    config = LocalHydraConfiguration('/home/ray/projects/helicopter/configs/atomic/flight_visualizer.yaml')
    visualizer_partial: partial = FlightVisualizer.from_hydra_configuration(config)
    visualizer: FlightVisualizer = visualizer_partial(aircraft_sm=aircraft_sm, kill_signal=kill_event)

    marker_queue = Queue()
    origin_queue = Queue()

    vis_thread = Thread(
        target=thread_runner,
        args=(visualizer.initialize, {
            'marker_queue': marker_queue,
            'origin_queue': origin_queue,
            'aircraft_lock': lock
        }),
        daemon=True
    )
    vis_thread.start()

    mock_markers = {
        0: {'rotation': Rotation.from_matrix(np.array([[    0.26356,     0.95562,    -0.13159],
                            [  -0.071074,     -0.1168,    -0.99061],
                            [   -0.96202,     0.27044,    0.037135]])), 'position': np.array([1.5395,  0.23548, -0.50982])},
    }
    marker_queue.put(mock_markers)

    time.sleep(1.0)

    extracted_origins = origin_queue.get(timeout=5.0)
    print(f"Received intermediate origins: {extracted_origins}")

    tracker_origin_data = {
        'camera_quat': Rotation.from_rotvec(np.array([0.0019705, 0.26787, -0])),
        'origin_quat': Rotation.from_rotvec(np.array([-0.22452, 0.20253, -1.6772])),
        'origin_position': np.array([1.9577, -0.15717, -0.89866])
    }
    origin_queue.put(tracker_origin_data)

    vis_thread.join(timeout=5.0)

    print('breakpoint')

    if vis_thread.is_alive():
        print("Error: Visualizer thread timed out.")
    else:
        print("Visualizer successfully initialized in isolation.")

    visualizer.cleanup()