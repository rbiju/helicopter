from collections.abc import MutableMapping
import queue
import threading
import time
from multiprocessing.synchronize import Lock, Event
from multiprocessing.shared_memory import SharedMemory

import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation
import pyrealsense2 as rs

from helicopter.configuration import HydraConfigurable
from helicopter.aircraft import Aircraft
from helicopter.vision import D435i, ErrorStateSquareRootUnscentedKalmanFilter
from helicopter.utils import PointQueue, Profiler, CommandBufferConstants

from .point_handler import TrackingPointHandler
from .filter_functions import propagate, transition_fn, measurement_fn, compose_fn


@HydraConfigurable
class Tracker:
    def __init__(self, aircraft_sm: SharedMemory,
                 point_handler: TrackingPointHandler,
                 camera: D435i,
                 ukf: ErrorStateSquareRootUnscentedKalmanFilter,
                 kill_signal: Event,
                 simulation_fps: float = 250.):
        self.aircraft_buffer = np.ndarray(shape=(Aircraft.N,),
                                          dtype=Aircraft.dtype,
                                          buffer=aircraft_sm.buf)

        self.point_handler = point_handler
        self.camera = camera
        self.ukf = ukf

        self.vision_thread = threading.Thread(target=self.vision_loop, daemon=True)
        self.vision_queue = queue.Queue(maxsize=5)

        self.camera_quat: Rotation = Rotation.from_rotvec(np.array([0, 0, 0.0]))
        self.initialized = False
        self.is_running = False

        self.profiler = Profiler()

        self.last_simulated_time = 0.0
        self.simulation_fps = simulation_fps

        self.first_frame = False
        self.first_frame_time = 0.0

        self.kill_signal = kill_signal

    @staticmethod
    def pack_intrinsics(intrinsics: rs.intrinsics) -> dict:
        return {
            'width': intrinsics.width,
            'height': intrinsics.height,
            'ppx': intrinsics.ppx,
            'ppy': intrinsics.ppy,
            'fx': intrinsics.fx,
            'fy': intrinsics.fy,
            'model': intrinsics.model.value,
            'coeffs': intrinsics.coeffs
        }

    def vision_loop(self):
        while self.is_running:
            frames = self.camera.pipeline.wait_for_frames()

            depth_image, ts_depth, ir_image, ts_ir, laser_state = self.camera.process_frames(frames)

            if ir_image is not None:
                try:
                    self.vision_queue.put((ts_depth, ir_image, depth_image), block=False)
                except queue.Full:
                    self.vision_queue.get_nowait()
                    self.vision_queue.put((ts_depth, ir_image, depth_image), block=False)


    def initialize(self, quat_sm: SharedMemory,
                   image_sm: SharedMemory,
                   intrinsics: MutableMapping):
        self.camera.start()

        intrinsics_dict = self.pack_intrinsics(self.camera.intrinsics)
        intrinsics.update(intrinsics_dict)
        camera_orientation_iters = 1000
        counter = 0
        accel_queue = PointQueue(maxlen=camera_orientation_iters)
        while counter < camera_orientation_iters:
            imu_frames = self.camera.imu_pipeline.wait_for_frames()
            accel, _, _, _ = self.camera.process_imu_frames(imu_frames)
            accel_queue.enqueue(accel)

        v_B = accel_queue.mean()
        v_W = np.array([0, 0, 1.0])

        self.camera_quat = Rotation.from_rotvec(np.cross(v_B, v_W))
        camera_quat = self.camera_quat.as_quat(canonical=True)
        buffer = np.ndarray(camera_quat.shape, dtype=camera_quat.dtype, buffer=quat_sm.buf)
        np.copyto(buffer, camera_quat)

        orientation_iters = 200
        print("Initializing helicopter orientation. Do not move aircraft.")
        counter = 0
        bad_frames = 0
        first_frame_collected = False
        while counter < orientation_iters:
            if bad_frames > 50:
                raise RuntimeError('Helicopter points not visible in 25% of init frames. Check that aircraft is in view.')
            frames = self.camera.pipeline.wait_for_frames()
            depth_image, ts_depth, ir_image, ts_ir, laser_state = self.camera.process_frames(frames)
            measure_out = self.point_handler.get_measured_points(ir_frame=ir_image,
                                                                 depth_frame=depth_image,
                                                                 intrinsics=self.camera.intrinsics)
            if measure_out is not None:
                if not first_frame_collected:
                    first_frame_buffer = np.ndarray(ir_image.shape, dtype=ir_image.dtype, buffer=image_sm.buf)
                    np.copyto(first_frame_buffer, ir_image)
                    first_frame_collected = True

                marker_coords, keypoints = measure_out
                self.point_handler.register_points(marker_coords)
            else:
                bad_frames += 1
                continue
            counter += 1

            r, t = self.point_handler.matcher.get_alignment(self.point_handler.init_points_coords)

            init_aircraft = Aircraft()
            init_aircraft.set_quaternion(r)
            init_aircraft.set_position(t)
            initial_state = init_aircraft.get_state_vector()

            np.copyto(self.aircraft_buffer, initial_state)

    def loop(self, command_sm: SharedMemory, lock: Lock, aircraft_lock: Lock):
        command_buffer = np.ndarray(shape=(CommandBufferConstants.N,),
                                    dtype=CommandBufferConstants.dtype,
                                    buffer=command_sm.buf)
        self.is_running = True
        self.vision_thread.start()

        while self.is_running:
            if self.kill_signal.is_set():
                raise RuntimeError('Tracker detected kill signal. Shutting down.')

            try:
                vision_time, ir_frame, depth_frame = self.vision_queue.get(timeout=0.05)
                vision_time = float(vision_time)

                if not self.first_frame:
                    self.first_frame = True
                    self.first_frame_time = vision_time

            except queue.Empty:
                time.sleep(0.001)
                continue

            self.profiler.start('Simulation')
            step_size = 1 / self.simulation_fps
            while self.last_simulated_time < vision_time:
                self.last_simulated_time += step_size

                commands = np.empty_like(command_buffer)
                with lock:
                    np.copyto(commands, command_buffer)

                aircraft = Aircraft.from_shared_memory_buffer(self.aircraft_buffer, aircraft_lock)
                nominal_state = jnp.array(aircraft.get_state_vector())

                propagated_nominal = propagate(nominal_state, step_size, commands)

                self.profiler.start("UKF_Predict")
                self.ukf = self.ukf.predict(transition_fn=transition_fn,
                                            dt=step_size,
                                            nominal_state=nominal_state,
                                            propagated_nominal=propagated_nominal,
                                            commands=commands)
                self.profiler.end("UKF_Predict")
            self.profiler.end('Simulation')

            self.profiler.start('Detection')
            measured_out = self.point_handler.get_measured_points(ir_frame=ir_frame,
                                                                  depth_frame=depth_frame,
                                                                  intrinsics=self.camera.intrinsics)
            self.profiler.end("Detection")

            if measured_out is None:
                aircraft = Aircraft.from_shared_memory_buffer(self.aircraft_buffer, aircraft_lock)
                current_nominal_state = jnp.array(aircraft.get_state_vector())

                nominal_state = np.array(compose_fn(current_nominal_state, self.ukf.x))
                self.ukf = self.ukf.reset()

                with aircraft_lock:
                    np.copyto(self.aircraft_buffer, nominal_state)
                continue
            else:
                measured_points, keypoints = measured_out
                # noinspection PyTypeChecker
                measured_points: np.ndarray = self.camera_quat.apply(measured_points)

                aircraft = Aircraft.from_shared_memory_buffer(self.aircraft_buffer, aircraft_lock)
                q = aircraft.quaternion
                t = aircraft.position.copy()
                nominal_state = jnp.array(aircraft.get_state_vector())

                self.profiler.start('Match_Points')

                measure_idx, ref_idx = self.point_handler.get_point_correspondence(q, t, measured_points)
                self.profiler.end('Match_Points')

                self.profiler.start('UKF_Update')
                z_points = measured_points[measure_idx]
                ref_points = self.point_handler.matcher.reference_points[ref_idx]

                for z_point, ref_point in zip(z_points, ref_points):
                    self.ukf = self.ukf.update(measurement_fn=measurement_fn,
                                               z_point=z_point,
                                               ref_point=ref_point,
                                               nominal_state=nominal_state)

                nominal_state = np.array(compose_fn(nominal_state,
                                                    self.ukf.x))

                self.ukf = self.ukf.reset()

                with aircraft_lock:
                    np.copyto(self.aircraft_buffer, nominal_state)
                self.profiler.end('UKF_Update')

    def cleanup(self):
        self.is_running = False

        if self.vision_thread.is_alive():
            self.vision_thread.join(timeout=1.0)
        self.camera.stop()
