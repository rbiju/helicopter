from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing.synchronize import Event, Lock
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
import time

import numpy as np

from helicopter.aircraft import Aircraft
from helicopter.configuration import HydraConfigurable, LocalHydraConfiguration
from helicopter.orchestration import FlightConductor
from helicopter.vision.tracking import Tracker
from helicopter.visualize import FlightVisualizer

from .base import Task


@dataclass
class MPContext:
    aircraft_sm: SharedMemory
    configuration_path: str
    start_event: Event
    shutdown_event: Event
    command_sm: SharedMemory
    marker_queue: mp.Queue
    origin_queue: mp.Queue
    aircraft_lock: Lock
    shared_memory_lock: Lock
    kill_signal: Event


def control_wrapper(ctx: MPContext):
    config = LocalHydraConfiguration(file_path=ctx.configuration_path)
    # noinspection PyUnresolvedReferences
    conductor = FlightConductor.from_hydra_configuration(config)

    # noinspection PyCallingNonCallable
    conductor: FlightConductor = conductor(aircraft_sm=ctx.aircraft_sm,
                                           kill_signal=ctx.kill_signal)

    try:
        conductor.initialize(aircraft_lock=ctx.aircraft_lock)
        ctx.start_event.set()
        conductor.loop(command_sm=ctx.command_sm,
                       lock=ctx.shared_memory_lock)
    finally:
        ctx.shutdown_event.set()
        conductor.cleanup()


def vision_wrapper(ctx: MPContext, ready_event: Event):
    config = LocalHydraConfiguration(file_path=ctx.configuration_path)
    # noinspection PyUnresolvedReferences
    tracker = Tracker.from_hydra_configuration(config)
    # noinspection PyCallingNonCallable
    tracker: Tracker = tracker(aircraft_sm=ctx.aircraft_sm,
                               kill_signal=ctx.kill_signal)

    try:
        tracker.initialize(marker_queue=ctx.marker_queue,
                           origin_queue=ctx.origin_queue,
                           aircraft_lock=ctx.aircraft_lock, )
        ready_event.set()
        ctx.start_event.wait()
        tracker.loop(command_sm=ctx.command_sm,
                     lock=ctx.shared_memory_lock)
    finally:
        ctx.shutdown_event.set()
        tracker.cleanup()


    while not ctx.shutdown_event.is_set():
        time.sleep(0.1)


def render_wrapper(ctx: MPContext, ready_event: Event):
    config = LocalHydraConfiguration(file_path=ctx.configuration_path)
    # noinspection PyUnresolvedReferences
    visualizer = FlightVisualizer.from_hydra_configuration(config)
    # noinspection PyCallingNonCallable
    visualizer: FlightVisualizer = visualizer(aircraft_sm=ctx.aircraft_sm,
                                              kill_signal=ctx.kill_signal)

    try:
        visualizer.initialize(marker_queue=ctx.marker_queue,
                              origin_queue=ctx.origin_queue,
                              aircraft_lock=ctx.aircraft_lock)
        ready_event.set()
        ctx.start_event.wait()
        visualizer.loop()
    finally:
        ctx.shutdown_event.set()
        visualizer.cleanup()


@HydraConfigurable
class Fly(Task):
    def __init__(self, image_dimension: list[int]):
        super().__init__()
        self.image_dimension = image_dimension

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
        vision_ready = mp.Event()
        render_ready = mp.Event()

        with SharedMemoryManager() as smm:
            aircraft_dummy = np.zeros(shape=(Aircraft.N,), dtype=np.float64)
            aircraft_sm = smm.SharedMemory(size=aircraft_dummy.nbytes)
            dummy_command = np.zeros(4, dtype=np.float64)
            command_sm = smm.SharedMemory(size=dummy_command.nbytes)
            sm_lock = mp.Lock()
            aircraft_lock = mp.Lock()
            kill_signal = mp.Event()
            marker_queue = mp.Queue()
            origin_queue = mp.Queue()

            ctx = MPContext(aircraft_sm=aircraft_sm,
                            configuration_path=configuration_path,
                            start_event=start_event,
                            shutdown_event=shutdown_event,
                            command_sm=command_sm,
                            marker_queue=marker_queue,
                            origin_queue=origin_queue,
                            aircraft_lock=aircraft_lock,
                            shared_memory_lock=sm_lock,
                            kill_signal=kill_signal)


            vision_process = mp.Process(target=vision_wrapper,
                                  args=(ctx, vision_ready))
            render_process = mp.Process(target=render_wrapper,
                                        args=(ctx, render_ready))

            vision_process.start()
            render_process.start()

            vision_ready.wait()
            render_ready.wait()

            try:
                print("Starting process loops")
                control_wrapper(ctx=ctx)

            except KeyboardInterrupt:
                print("\nCtrl+C detected. Shutting down...")

            finally:
                ctx.shutdown_event.set()
                ctx.kill_signal.set()

                vision_process.join(timeout=5)
                if vision_process.is_alive():
                    vision_process.terminate()

                render_process.join(timeout=5)
                if render_process.is_alive():
                    render_process.terminate()
