import logging
import time
import threading
import queue

import numpy as np
import jax
import jax.numpy as jnp
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from helicopter.utils import PointQueue, Profiler
from helicopter.vision import D435i
from helicopter.vision import ErrorStateSquareRootUnscentedKalmanFilter as UKF
from helicopter.vision.point_detection import PointHandler

from .logger import StateLogger
from .camera_state_handler import CameraStateHandler

from .filter_functions import compose_fn, propagate, transition_fn, measurement_fn

logging.basicConfig(level=logging.WARNING)


class Scanner:
    def __init__(self,
                 device: D435i,
                 point_handler: PointHandler,
                 camera_state_handler: CameraStateHandler,
                 ukf: UKF,
                 measurement_time: float = 5.0):
        self.device = device
        self.point_handler = point_handler
        self.camera_state_handler = camera_state_handler

        self.ukf = ukf
        self.init_jax()

        self.profiler = Profiler()

        self.measurement_time = measurement_time

        self.is_running = False
        self.started_running = False

        self.start_time = 0.0

        self.imu_thread = threading.Thread(target=self.imu_loop, daemon=True)
        self.vision_thread = threading.Thread(target=self.vision_loop, daemon=True)
        self.lock = threading.Lock()

        self.logger = StateLogger()

        self.last_fused_time = 0.0
        self.first_imu_frame = True

        self.vision_queue = queue.Queue(maxsize=1)
        self.imu_queue = queue.Queue(maxsize=25)

        self.cleaned_up = False

    def init_jax(self):
        print("Compiling JAX kernels...")

        dummy_dt = jnp.array(0.01, dtype=jnp.float32)
        dummy_nominal = jnp.array(self.camera_state_handler.nominal_state, dtype=jnp.float32)
        dummy_error = jnp.zeros(15, dtype=jnp.float32)

        accel = jnp.array(np.zeros(3, dtype=np.float32))
        gyro = jnp.array(np.zeros(3, dtype=np.float32))
        g_world = jnp.array(self.camera_state_handler.g, dtype=jnp.float32)

        _ = propagate(dummy_nominal, dummy_dt, accel, gyro, g_world)
        _ = compose_fn(dummy_nominal, dummy_error)

        tmp = self.ukf.predict(transition_fn=transition_fn,
                               dt=dummy_dt,
                               nominal_state=dummy_nominal,
                               propagated_nominal=dummy_nominal,
                               accel=accel,
                               gyro=gyro,
                               g_world=g_world)

        dummy_points = np.ones((10, 3), dtype=np.float32)

        for z_point, ref_point in zip(dummy_points, dummy_points):
            _ = self.ukf.update(measurement_fn=measurement_fn,
                                z_point=z_point,
                                ref_point=ref_point,
                                nominal_state=dummy_nominal)

        jax.block_until_ready(tmp)

        print("JAX Compilation complete")

    def initialize_orientation(self):
        accel_queue = PointQueue(750, np.array([0, 0, 0.0]))
        gyro_queue = PointQueue(750, np.array([0, 0, 0.0]))

        orientation_iters = 500
        pbar = tqdm(total=orientation_iters, desc="Initializing sensor orientation. Do not move camera")
        counter = 0
        while counter < orientation_iters:
            imu_frames = self.device.imu_pipeline.wait_for_frames()
            imu_data = self.device.process_imu_frames(imu_frames)
            if imu_data is not None:
                accel_data, ts_accel, gyro_data, ts_gyro = imu_data
                accel_queue.enqueue(accel_data)
                gyro_queue.enqueue(gyro_data)
                pbar.update(1)
                counter += 1

        v_B = accel_queue.mean()
        v_B_norm = np.linalg.norm(v_B)
        v_B_unit = v_B / v_B_norm

        v_W_unit = np.array([0.0, 0.0, 1.0])

        rotation_axis = np.cross(v_B_unit, v_W_unit)
        axis_norm = np.linalg.norm(rotation_axis)
        if axis_norm > 1e-6:
            rotation_axis /= axis_norm
            rotation_angle = np.arccos(np.dot(v_B_unit, v_W_unit))
        else:
            rotation_angle = 0.0

        theta_half = rotation_angle / 2.0

        quat_array = np.array((*(np.sin(theta_half) * rotation_axis), np.cos(theta_half)))
        quat = Rotation.from_quat(quat_array)

        expected_gravity_body = quat.inv().apply(self.camera_state_handler.g)
        self.camera_state_handler.accelerometer_bias = v_B - expected_gravity_body

        self.camera_state_handler.gyro_bias = gyro_queue.mean()
        self.camera_state_handler.quaternion = quat

        print(f"Orientation initialized, \n "
              f"mean acceleration: {v_B} \n"
              f"mean gyro: {gyro_queue.mean()}")

    def imu_loop(self):
        is_first_frame = True
        while self.is_running:
            try:
                imu_frames = self.device.imu_pipeline.wait_for_frames(timeout_ms=10)
            except RuntimeError:
                continue

            imu_data = self.device.process_imu_frames(imu_frames)

            if imu_data is None:
                continue

            accel, _, gyro, timestamp = imu_data

            if is_first_frame:
                is_first_frame = False
                continue

            dt = timestamp - self.device.last_imu_time
            if dt <= 0:
                continue

            try:
                self.imu_queue.put((timestamp, accel, gyro), block=False)
            except queue.Full:
                continue

    def vision_loop(self):
        while self.is_running:
            try:
                frames = self.device.pipeline.wait_for_frames(timeout_ms=20)
            except RuntimeError:
                continue

            depth_image, ts_depth, ir_image, ts_ir, laser_state = self.device.process_frames(frames)

            if ir_image is not None:
                try:
                    self.vision_queue.put((ts_ir, ir_image, depth_image), block=False)
                except queue.Full:
                    continue

    def loop(self):
        self.device.time_queue.clear()

        self.started_running = True
        self.is_running = True
        self.imu_thread.start()
        self.vision_thread.start()

        pbar = tqdm(desc=f"Starting measurement lasting {self.measurement_time} seconds")
        self.start_time = time.time()
        while self.is_running:
            pbar.update(1)
            elapsed_time = time.time() - self.start_time
            if elapsed_time > self.measurement_time:
                self.is_running = False

            try:
                vision_data = self.vision_queue.get(timeout=0.1)
                vision_time, ir_frame, depth_frame = vision_data
            except queue.Empty:
                continue

            self.profiler.start("E2E")
            self.profiler.start("Measure_Points")
            measured_out = self.point_handler.get_measured_points(ir_frame, depth_frame, self.device.intrinsics)
            self.profiler.end("Measure_Points")
            if measured_out is None:
                continue
            else:
                measured_points, keypoints = measured_out

            self.profiler.start('IMU_Integration')
            while not self.imu_queue.empty():
                with self.imu_queue.mutex:
                    imu_t = self.imu_queue.queue[0][0]

                if imu_t > vision_time:
                    break

                imu_t, accel, gyro = self.imu_queue.get()

                if self.first_imu_frame:
                    self.last_fused_time = imu_t
                    self.first_imu_frame = False
                    continue

                self.logger.log_imu(timestamp=imu_t, accel=accel, gyro=gyro)
                accel = jnp.array(accel, dtype=jnp.float32)
                gyro = jnp.array(gyro, dtype=jnp.float32)
                g_world = jnp.array(self.camera_state_handler.g, dtype=jnp.float32)

                dt = jnp.array(imu_t - self.last_fused_time, dtype=jnp.float32)
                self.last_fused_time = imu_t

                nominal_state = jnp.array(self.camera_state_handler.nominal_state)
                propagated_nominal = propagate(nominal_state, dt, accel, gyro,
                                               g_world)

                self.profiler.start("UKF_Predict")
                self.ukf = self.ukf.predict(transition_fn=transition_fn,
                                            dt=dt,
                                            nominal_state=nominal_state,
                                            propagated_nominal=propagated_nominal,
                                            accel=accel,
                                            gyro=gyro,
                                            g_world=g_world)
                self.profiler.end("UKF_Predict")

                propagated_nominal = np.array(propagated_nominal)

                self.logger.log_state(timestamp=imu_t, event='imu', state_vector=propagated_nominal)
                self.camera_state_handler.set_state_from_nominal(propagated_nominal)
            self.profiler.end('IMU_Integration')

            camera_pos = self.camera_state_handler.position
            camera_quat = self.camera_state_handler.quaternion
            nominal_state = jnp.array(self.camera_state_handler.nominal_state)

            self.profiler.start('Match_Points')
            measured_points_wf, measure_idx, reference_idx = self.point_handler.match_points(
                measured_points,
                camera_position=camera_pos,
                camera_quat=camera_quat)

            self.profiler.end('Match_Points')

            self.profiler.start('UKF_Update')
            z_points = measured_points[measure_idx]
            ref_points = self.point_handler.point_map[reference_idx]

            for z_point, ref_point in zip(z_points, ref_points):
                self.ukf = self.ukf.update(measurement_fn=measurement_fn,
                                           z_point=z_point,
                                           ref_point=ref_point,
                                           nominal_state=nominal_state)

            nominal_state = np.array(compose_fn(nominal_state,
                                                self.ukf.x))

            self.ukf = self.ukf.reset()

            self.camera_state_handler.set_state_from_nominal(nominal_state)
            self.profiler.end('UKF_Update')

            self.point_handler.append_points(measured_points_wf, reference_idx)

            self.logger.log_state(timestamp=vision_time, event='vision', state_vector=nominal_state)
            self.profiler.end("E2E")
        pbar.close()

    def cleanup(self):
        print("Cleaning up sensor")

        self.is_running = False

        if self.started_running:
            if self.imu_thread.is_alive():
                self.imu_thread.join(timeout=1.0)

            if self.vision_thread.is_alive():
                self.vision_thread.join(timeout=1.0)

            self.logger.save()
            print(self.profiler)

        self.device.stop()
        self.cleaned_up = True

    def scan(self):
        try:
            self.device.start()
            self.initialize_orientation()
            with jax.log_compiles():
                self.loop()

        finally:
            if not self.cleaned_up:
                self.cleanup()
