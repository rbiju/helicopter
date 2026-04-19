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
    orientation_ready: Event
    kill_signal: Event
    init_only: bool = False


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
        if ctx.init_only:
            print('Initialization completed. Waiting for 30 seconds...')
            time.sleep(30.0)
        else:
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
                           orientation_ready=ctx.orientation_ready,
                           aircraft_lock=ctx.aircraft_lock, )
        ready_event.set()
        ctx.start_event.wait()
        if ctx.init_only:
            time.sleep(30.0)
        else:
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
                              orientation_ready=ctx.orientation_ready,
                              aircraft_lock=ctx.aircraft_lock)
        ready_event.set()
        ctx.start_event.wait()
        if ctx.init_only:
            time.sleep(30.0)
        else:
            visualizer.loop(command_sm=ctx.command_sm,
                            lock=ctx.shared_memory_lock)
    finally:
        ctx.shutdown_event.set()
        visualizer.cleanup()


@HydraConfigurable
class Fly(Task):
    def __init__(self, init_only: bool = False):
        super().__init__()
        self.init_only = init_only

    def run(self, configuration_path: str):
        mp.set_start_method('spawn')

        start_event = mp.Event()
        shutdown_event = mp.Event()
        vision_ready = mp.Event()
        render_ready = mp.Event()
        orientation_ready = mp.Event()

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
                            kill_signal=kill_signal,
                            orientation_ready=orientation_ready,
                            init_only=self.init_only,)


            vision_process = mp.Process(target=vision_wrapper,
                                  args=(ctx, vision_ready))
            render_process = mp.Process(target=render_wrapper,
                                        args=(ctx, render_ready))

            vision_process.start()
            render_process.start()

            vision_ready.wait()
            render_ready.wait()

            try:
                print("Starting control process")
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
