import queue
import threading
import time
from multiprocessing.synchronize import Event, Lock
from multiprocessing.shared_memory import SharedMemory

import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

from helicopter.configuration import HydraConfigurable
from helicopter.aircraft import Aircraft, FlightStates
from helicopter.vision import D435i, ErrorStateSquareRootUnscentedKalmanFilter
from helicopter.utils import PointQueue, Profiler

from .point_handler import TrackingPointHandler
from .filter_functions import propagate, transition_fn, measurement_fn, compose_fn


@HydraConfigurable
class Tracker:
    def __init__(self, aircraft: Aircraft,
                 lock: Lock,
                 point_handler: TrackingPointHandler,
                 camera: D435i,
                 ukf: ErrorStateSquareRootUnscentedKalmanFilter,
                 simulation_fps: float = 250.):
        self.aircraft = aircraft
        self.lock = lock

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


    def initialize(self, camera_quat_sm: SharedMemory,
                 init_image_sm: SharedMemory,
                 quat_flag: Event,
                 init_image_flag: Event,
                 init_flag: Event,):
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
        buffer = np.ndarray(camera_quat.shape, dtype=camera_quat.dtype, buffer=camera_quat_sm.buf)
        buffer[:] = camera_quat[:]
        quat_flag.set()

        self.aircraft.set_flight_state(FlightStates.IDLE)

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
                    first_frame_buffer = np.ndarray(ir_image.shape, dtype=ir_image.dtype, buffer=init_image_sm.buf)
                    first_frame_buffer[:] = ir_image[:]
                    init_image_flag.set()
                    first_frame_collected = True

                marker_coords, keypoints = measure_out
                self.point_handler.register_points(marker_coords)
            else:
                bad_frames += 1
                continue
            counter += 1

            r, t = self.point_handler.matcher.get_alignment(self.point_handler.init_points_coords)

            with self.lock:
                self.aircraft.set_quaternion(r)
                self.aircraft.set_position(t)

            init_flag.set()

    def loop(self, command_buffer: SharedMemory):
        self.is_running = True
        self.vision_thread.start()

        while self.is_running:
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

                with self.lock:
                    commands = np.ndarray((4,), dtype=np.int64, buffer=command_buffer.buf).copy()

                with self.lock:
                    nominal_state = jnp.array(self.aircraft.get_state_vector())

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

            if measured_out is not None:
                measured_points, keypoints = measured_out
            else:
                nominal_state = jnp.array(self.aircraft.get_state_vector())
                nominal_state = np.array(compose_fn(nominal_state, self.ukf.x))
                self.ukf = self.ukf.reset()
                self.aircraft.set_state_vector(nominal_state)
                continue

            # Perform ICP + UKF Update here
            self.profiler.start('Match_Points')
            with self.lock:
                q = self.aircraft.quaternion
                t = self.aircraft.position.copy()

            ref_idx, measure_idx = self.point_handler.get_point_correspondence(q, t, measured_points)
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

            self.aircraft.set_state_vector(nominal_state)
            self.profiler.end('UKF_Update')


