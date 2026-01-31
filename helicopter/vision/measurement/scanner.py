import time
import threading

from tqdm import tqdm
import numpy as np
import quaternion
from filterpy.kalman import UnscentedKalmanFilter

from helicopter.vision.d435i import D435i
from helicopter.vision.point_detection.measurement.point_handler import PointHandler
from .madgwick import MadgwickFilter
from .logger import StateLogger
from .camera_state_handler import CameraStateHandler
from .utils import PointQueue

from .filter_functions import compose_fn, project_to_tangent_space, propagate


class Scanner:
    def __init__(self,
                 device: D435i,
                 point_handler: PointHandler,
                 camera_state_handler: CameraStateHandler,
                 ukf: UnscentedKalmanFilter,
                 madgwick: MadgwickFilter,
                 measurement_time: float = 5.0):
        self.device = device
        self.point_handler = point_handler
        self.camera_state_handler = camera_state_handler
        self.ukf = ukf
        self.madgwick = madgwick

        self.measurement_time = measurement_time

        self.ukf_Q = ukf.Q.copy()

        self.is_running = False
        self.started_running = False

        self.timer = 0.0

        self.state_lock = threading.Lock()
        self.imu_thread = threading.Thread(target=self.imu_loop, daemon=True)

        self.logger = StateLogger()

        self.max_dt = 0.0
        self.detect_time = 0

    def initialize_orientation(self):
        accel_queue = PointQueue(500, np.array([0, 0, 0.0]))
        gyro_queue = PointQueue(500, np.array([0, 0, 0.0]))

        orientation_iters = 1500
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

        quat = quaternion.quaternion(
            np.cos(theta_half),
            *(np.sin(theta_half) * rotation_axis)
        )

        self.camera_state_handler.accelerometer_bias = ((quat * quaternion.quaternion(0,
                                                                                      *v_B) * quat.inverse()).imag -
                                                        self.camera_state_handler.g)
        self.camera_state_handler.gyro_bias = gyro_queue.mean()
        self.camera_state_handler.quaternion = quat
        self.madgwick.q = quaternion.as_float_array(quat)

        self.camera_state_handler.last_quaternion = quat

        print("Orientation initialized")

    def imu_loop(self):
        is_first_frame = True
        while self.is_running:
            imu_frames = self.device.imu_pipeline.wait_for_frames()

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

            self.logger.log_imu(timestamp=timestamp, accel=accel, gyro=gyro)

            if dt > self.max_dt:
                self.max_dt = dt

            scale_factor = dt / (1 / 200)

            lower_bound = 9.0
            upper_bound = 10.6

            if lower_bound < np.linalg.norm(accel) < upper_bound:
                current_beta = 0.033
            else:
                current_beta = 0.0

            q_madgwick = self.madgwick.update(accel=accel, gyro=gyro, dt=dt, beta=current_beta)

            with self.state_lock:
                nominal_state = self.camera_state_handler.nominal_state.copy()
                propagated_nominal = propagate(nominal_state, dt, accel, gyro,
                                               self.camera_state_handler.g,
                                               q_madgwick)

                self.logger.log_state(timestamp=timestamp, event='imu', state_vector=propagated_nominal)
                self.ukf.Q = self.ukf_Q * scale_factor
                self.ukf.predict(dt=dt,
                                 nominal_state=self.camera_state_handler.nominal_state.copy(),
                                 propagated_nominal=propagated_nominal,
                                 accelerometer=accel,
                                 gyro=gyro,
                                 g_world=self.camera_state_handler.g,
                                 q_madgwick=q_madgwick)

                self.camera_state_handler.set_state_from_nominal(propagated_nominal)
                self.ukf.x.fill(0)

    def loop(self):
        print("Starting measurement in...")
        time.sleep(0.5)
        print('3')
        time.sleep(0.5)
        print('2')
        time.sleep(0.5)
        print('1')

        # flushing camera buffer
        for _ in range(10):
            self.device.pipeline.poll_for_frames()

        while True:
            success, _ = self.device.imu_pipeline.try_wait_for_frames(timeout_ms=10)
            if not success:
                break

        self.device.time_queue.clear()

        self.started_running = True
        self.is_running = True

        self.imu_thread.start()
        self.timer = time.time()
        while self.is_running:
            elapsed_time = time.time() - self.timer
            if elapsed_time > self.measurement_time:
                self.is_running = False

            frames = self.device.pipeline.wait_for_frames()
            with self.state_lock:
                camera_pos = self.camera_state_handler.position.copy()
                camera_quat = self.camera_state_handler.quaternion.copy()

            depth_image, ts_depth, ir_image, ts_ir, laser_state = self.device.process_frames(frames)
            if ir_image is not None:
                start_detect = time.perf_counter()
                measured_out = self.point_handler.get_measured_points(ir_frame=ir_image,
                                                                      depth_frame=depth_image,
                                                                      intrinsics=self.device.intrinsics)
                if measured_out is None:
                    print('Not enough points')
                    continue
                else:
                    measured_points_cf, keypoints = measured_out

                    measured_points_wf, measure_idx, reference_idx = self.point_handler.match_points(
                        measured_points_cf,
                        camera_position=self.camera_state_handler.last_position,
                        camera_quat=self.camera_state_handler.last_quaternion)

                    reference_points = self.point_handler.point_map[reference_idx]

                    success, visual_quat, visual_translation = self.camera_state_handler.get_visual_pose(
                        measured_points=measured_points_cf[measure_idx],
                        reference_points=reference_points,
                        quat=camera_quat)

                    self.camera_state_handler.last_quaternion = visual_quat
                    self.camera_state_handler.last_position = visual_translation

                    if not success:
                        continue
                    else:
                        position_delta = visual_translation - camera_pos
                        if np.linalg.norm(position_delta) > 0.05:
                            print(
                                f'Jump from {camera_pos} to {visual_translation} detected')
                            continue

                        self.point_handler.append_points(measured_points_wf, reference_idx)

                        projected_z = project_to_tangent_space(visual_quat, visual_translation,
                                                               camera_quat, camera_pos)

                        with self.state_lock:
                            self.ukf.update(
                                z=projected_z,
                                nominal_state=self.camera_state_handler.nominal_state.copy())

                            nominal_state = compose_fn(self.camera_state_handler.nominal_state.copy(),
                                                       self.ukf.x.copy())

                            self.logger.log_state(timestamp=ts_ir, event='vision', state_vector=nominal_state)
                            self.logger.log_vision(timestamp=ts_ir, quat=visual_quat, translation=visual_translation)

                            self.camera_state_handler.set_state_from_nominal(nominal_state.copy())
                            self.ukf.x.fill(0)

                        end_detect = time.perf_counter()

                        self.detect_time = (end_detect - start_detect)

    def scan(self):
        try:
            self.initialize_orientation()
            self.loop()

        finally:
            print("Cleaning up sensor")
            self.device.stop()

            if self.is_running:
                self.is_running = False

            if self.started_running:
                self.logger.save()
