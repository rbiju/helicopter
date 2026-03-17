from collections.abc import MutableMapping
from dataclasses import dataclass
import multiprocessing as mp
from multiprocessing.synchronize import Event, Lock
from multiprocessing.shared_memory import SharedMemory
from multiprocessing.managers import SharedMemoryManager
import time

import numpy as np

from helicopter.aircraft import AircraftManager, Aircraft
from helicopter.configuration import HydraConfigurable, LocalHydraConfiguration
from helicopter.orchestration import FlightConductor
from helicopter.vision.tracking import Tracker
from helicopter.visualize import FlightVisualizer

from .base import Task


@dataclass
class MPCContext:
    aircraft_proxy: Aircraft
    configuration_path: str
    start_event: Event
    shutdown_event: Event
    image_sm: SharedMemory
    quat_sm: SharedMemory
    command_sm: SharedMemory
    image_dimension: list[int]
    vision_ready: Event
    intrinsics_dict: MutableMapping
    shared_memory_lock: Lock


def control_wrapper(ctx: MPCContext, ready_event: Event):
    config = LocalHydraConfiguration(file_path=ctx.configuration_path)
    # noinspection PyUnresolvedReferences
    conductor = FlightConductor.from_hydra_configuration(config, find_key=True)

    # noinspection PyCallingNonCallable
    conductor: FlightConductor = conductor(ctx.aircraft_proxy)

    try:
        conductor.initialize()
        ready_event.set()
        ctx.start_event.wait()
    finally:
        ctx.shutdown_event.set()
        conductor.cleanup()


def vision_wrapper(ctx: MPCContext, ready_event: Event):
    config = LocalHydraConfiguration(file_path=ctx.configuration_path)
    # noinspection PyUnresolvedReferences
    tracker = Tracker.from_hydra_configuration(config, find_key=True)
    # noinspection PyCallingNonCallable
    tracker: Tracker = tracker(ctx.aircraft_proxy)

    try:
        tracker.initialize(image_sm=ctx.image_sm,
                           quat_sm=ctx.quat_sm,
                           intrinsics=ctx.intrinsics_dict)
        ready_event.set()
        ctx.start_event.wait()
    finally:
        ctx.shutdown_event.set()
        tracker.cleanup()


    while not ctx.shutdown_event.is_set():
        time.sleep(0.1)


def render_wrapper(ctx: MPCContext, ready_event: Event):
    config = LocalHydraConfiguration(file_path=ctx.configuration_path)
    # noinspection PyUnresolvedReferences
    visualizer = FlightVisualizer.from_hydra_configuration(config, find_key=True)
    # noinspection PyCallingNonCallable
    visualizer: FlightVisualizer = visualizer(ctx.aircraft_proxy)

    ctx.vision_ready.wait()

    try:
        visualizer.initialize(ctx.image_sm, ctx.image_dimension, ctx.intrinsics_dict, ctx.quat_sm)
        ready_event.set()
        ctx.start_event.wait()
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
        control_ready = mp.Event()
        vision_ready = mp.Event()
        render_ready = mp.Event()

        with SharedMemoryManager() as smm:
            with AircraftManager() as manager:
                shared_aircraft = manager.Aircraft()
                image_dummy = np.zeros(self.image_dimension, dtype=np.uint8)
                image_sm = smm.SharedMemory(size=image_dummy.nbytes)
                quat_dummy = np.zeros(4, dtype=np.float64)
                quat_sm = smm.SharedMemory(size=quat_dummy.nbytes)
                shared_intrinsics_dict = manager.dict()

                ctx = MPCContext(aircraft_proxy=shared_aircraft,
                                 configuration_path=configuration_path,
                                 start_event=start_event,
                                 shutdown_event=shutdown_event,
                                 image_sm=image_sm,
                                 quat_sm=quat_sm,
                                 image_dimension=self.image_dimension,
                                 vision_ready=vision_ready,
                                 intrinsics_dict=shared_intrinsics_dict)


                control_process = mp.Process(target=control_wrapper,
                                       args=(ctx, control_ready))
                vision_process = mp.Process(target=vision_wrapper,
                                      args=(ctx, vision_ready))
                render_process = mp.Process(target=render_wrapper,
                                            args=(ctx, render_ready))

                control_process.start()
                vision_process.start()
                render_process.start()

                control_ready.wait()
                vision_ready.wait()
                render_ready.wait()

                start_event.set()

                try:
                    while True:
                        time.sleep(1)

                except KeyboardInterrupt:

                    shutdown_event.set()

                    control_process.join(timeout=5)
                    vision_process.join(timeout=5)
                    render_process.join(timeout=5)
