import multiprocessing as mp
from multiprocessing.managers import SharedMemoryManager
import time

from helicopter.configuration import HydraConfigurable, LocalHydraConfiguration
from helicopter.orchestration import FlightConductor
from helicopter.vision.tracking import Tracker
from helicopter.visualize import FlightVisualizer

from .base import Task

from helicopter.aircraft import AircraftManager


def control_wrapper(aircraft_proxy, configuration_path, ready_event, start_event, shutdown_event):
    config = LocalHydraConfiguration(file_path=configuration_path)
    # noinspection PyUnresolvedReferences
    conductor = FlightConductor.from_hydra_configuration(config, find_key=True)

    # noinspection PyCallingNonCallable
    conductor: FlightConductor = conductor(aircraft_proxy)
    conductor.initialize()
    ready_event.set()
    start_event.wait()

    while not shutdown_event.is_set():
        time.sleep(0.1)


def vision_wrapper(aircraft_proxy, configuration_path, ready_event, start_event, shutdown_event):
    config = LocalHydraConfiguration(file_path=configuration_path)
    # noinspection PyUnresolvedReferences
    tracker = Tracker.from_hydra_configuration(config, find_key=True)
    # noinspection PyCallingNonCallable
    tracker: Tracker = tracker(aircraft_proxy)
    ready_event.set()
    start_event.wait()

    while not shutdown_event.is_set():
        time.sleep(0.1)


def visualizer_wrapper(aircraft_proxy, configuration_path, ready_event, start_event, shutdown_event):
    config = LocalHydraConfiguration(file_path=configuration_path)
    # noinspection PyUnresolvedReferences
    visualizer = FlightVisualizer.from_hydra_configuration(config, find_key=True)
    # noinspection PyCallingNonCallable
    visualizer: FlightVisualizer = visualizer(aircraft_proxy)
    ready_event.set()
    start_event.wait()

    while not shutdown_event.is_set():
        time.sleep(0.1)


@HydraConfigurable
class Fly(Task):
    def __init__(self,
                 **kwargs):
        super().__init__(**kwargs)

    def run(self, configuration_path: str):
        """
        Sequence:

        1. Aircraft state manager gets made, passed to all three objects
        2. Vision thread captures first image, camera orientation during init, passes them to visualizer
        3. All three finish init
        4. Main loop starts
            a) Tracker starts to estimate state
            b) controller begins to calculate commands once rc is ready to send, grabs most recent state from aircraft
            c) controller sends most recent sent command to tracker
        5. On kill signal, all three enter cleanup and shut down

        """
        start_event = mp.Event()
        shutdown_event = mp.Event()
        control_ready = mp.Event()
        vision_ready = mp.Event()
        visualizer_ready = mp.Event()

        with SharedMemoryManager() as smm:
            with AircraftManager() as manager:
                shared_aircraft = manager.Aircraft()

                p_control = mp.Process(target=control_wrapper,
                                       args=(shared_aircraft, configuration_path, control_ready, start_event, shutdown_event))
                p_vision = mp.Process(target=vision_wrapper,
                                      args=(shared_aircraft, vision_ready, start_event, shutdown_event))
                p_visualizer = mp.Process(target=visualizer_wrapper,
                                          args=(shared_aircraft, visualizer_ready, start_event, shutdown_event))

                p_control.start()
                p_vision.start()
                p_visualizer.start()

                control_ready.wait()
                vision_ready.wait()
                visualizer_ready.wait()

                start_event.set()

                try:
                    while True:
                        time.sleep(1)

                except KeyboardInterrupt:

                    shutdown_event.set()

                    p_control.join(timeout=5)
                    p_vision.join(timeout=5)
                    p_visualizer.join(timeout=5)
