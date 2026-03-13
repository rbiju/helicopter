import multiprocessing as mp
import time

from helicopter.configuration import HydraConfigurable, LocalHydraConfiguration
from helicopter.orchestration import FlightConductor
from helicopter.vision.tracking import Tracker
from helicopter.visualize import FlightVisualizer

from .base import Task

from helicopter.aircraft import AircraftManager


def control_wrapper(aircraft_proxy, configuration_path, ready_event, start_event, shutdown_event):
    config = LocalHydraConfiguration(file_path=configuration_path)
    conductor = FlightConductor.from_hydra_configuration(config, find_key=True)
    conductor = conductor(aircraft_proxy)
    ready_event.set()
    start_event.wait()

    while not shutdown_event.is_set():
        time.sleep(0.1)


def vision_wrapper(aircraft_proxy, configuration_path, ready_event, start_event, shutdown_event):
    config = LocalHydraConfiguration(file_path=configuration_path)
    tracker = Tracker.from_hydra_configuration(config, find_key=True)
    tracker = tracker(aircraft_proxy)
    ready_event.set()
    start_event.wait()

    while not shutdown_event.is_set():
        time.sleep(0.1)


def visualizer_wrapper(aircraft_proxy, configuration_path, ready_event, start_event, shutdown_event):
    config = LocalHydraConfiguration(file_path=configuration_path)
    visualizer = FlightVisualizer.from_hydra_configuration(config, find_key=True)
    visualizer = visualizer(aircraft_proxy)
    ready_event.set()
    start_event.wait()

    while not shutdown_event.is_set():
        time.sleep(0.1)


@HydraConfigurable
class OrchestrateFlight(Task):
    def __init__(self,
                 project_name: str = "helicopter-v1",
                 **kwargs):
        super().__init__(**kwargs)
        self.project_name = project_name

    def run(self, configuration_path: str):
        start_event = mp.Event()
        shutdown_event = mp.Event()
        control_ready = mp.Event()
        vision_ready = mp.Event()
        visualizer_ready = mp.Event()

        print("[Main] Starting Aircraft state manager...")
        with AircraftManager() as manager:
            shared_aircraft = manager.Aircraft()

            p_control = mp.Process(target=control_wrapper,
                                   args=(shared_aircraft, control_ready, start_event, shutdown_event))
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
