import time
import threading
from collections import deque
import queue

from tqdm import tqdm
import numpy as np
import quaternion
from filterpy.kalman import UnscentedKalmanFilter

from helicopter.vision.d435i import D435i
from helicopter.vision.point_detection.measurement.point_handler import PointHandler
from .logger import StateLogger
from .camera_state_handler import CameraStateHandler
from helicopter.vision import PointQueue

from .filter_functions import compose_fn, project_to_tangent_space, propagate


class Scanner:
    def __init__(self,
                 device: D435i,
                 point_handler: PointHandler,
                 camera_state_handler: CameraStateHandler,
                 ukf: UnscentedKalmanFilter,
                 measurement_time: float = 5.0,
                 vis_ema: float = 0.5):
        self.device = device
        self.point_handler = point_handler
        self.camera_state_handler = camera_state_handler
        self.ukf = ukf

        self.measurement_time = measurement_time

        self.ukf_Q = ukf.Q.copy()
        self.GATE_THRESHOLD = 3.0

        self.is_running = False
        self.started_running = False

        self.start_time = 0.0

        self.imu_thread = threading.Thread(target=self.imu_loop, daemon=True)
        self.vision_thread = threading.Thread(target=self.vision_loop, daemon=True)

        self.logger = StateLogger()

        self.update_time = 0

        self.last_fused_time = 0.0
        self.first_imu_frame = True

        self.vision_queue = queue.Queue()
        self.imu_queue = deque(maxlen=25)

        self.vis_ema_param = vis_ema
        self.smooth_vis_translation = None
        self.smooth_vis_quat = None

    def initialize_orientation(self):
        accel_queue = PointQueue(750, np.array([0, 0, 0.0]))
        gyro_queue = PointQueue(750, np.array([0, 0, 0.0]))

        orientation_iters = 1000
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

            self.imu_queue.append((timestamp, accel, gyro))

    def vision_loop(self):
        while self.is_running:
            frames = self.device.pipeline.wait_for_frames()

            depth_image, ts_depth, ir_image, ts_ir, laser_state = self.device.process_frames(frames)

            if ir_image is not None:
                measured_out = self.point_handler.get_measured_points(ir_frame=ir_image,
                                                                      depth_frame=depth_image,
                                                                      intrinsics=self.device.intrinsics)

                if measured_out is None:
                    continue
                else:
                    measured_points, keypoints = measured_out
                    self.vision_queue.put((ts_ir, measured_points))

    def loop(self):
        # flushing device buffers
        for _ in range(10):
            self.device.pipeline.poll_for_frames()

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
                vision_time, measured_points = vision_data
            except queue.Empty:
                continue

            start_update = time.perf_counter()

            while self.imu_queue:
                imu_t, accel, gyro = self.imu_queue[0]

                if self.first_imu_frame:
                    self.last_fused_time = imu_t
                    self.first_imu_frame = False
                    continue

                self.logger.log_imu(timestamp=imu_t, accel=accel, gyro=gyro)

                if imu_t > vision_time:
                    break

                self.imu_queue.popleft()
                dt = imu_t - self.last_fused_time
                self.last_fused_time = imu_t

                scale_factor = dt / (1 / 200)

                nominal_state = self.camera_state_handler.nominal_state.copy()
                propagated_nominal = propagate(nominal_state, dt, accel, gyro,
                                               self.camera_state_handler.g)

                self.ukf.Q = self.ukf_Q * scale_factor
                self.ukf.predict(dt=dt,
                                 nominal_state=self.camera_state_handler.nominal_state.copy(),
                                 propagated_nominal=propagated_nominal,
                                 accelerometer=accel,
                                 gyro=gyro,
                                 g_world=self.camera_state_handler.g)

                self.logger.log_state(timestamp=imu_t, event='imu', state_vector=propagated_nominal)
                self.camera_state_handler.set_state_from_nominal(propagated_nominal)

            camera_pos = self.camera_state_handler.position
            camera_quat = self.camera_state_handler.quaternion

            measured_points_wf, measure_idx, reference_idx = self.point_handler.match_points(
                measured_points,
                camera_position=camera_pos,
                camera_quat=camera_quat)

            reference_points = self.point_handler.point_map[reference_idx]

            success, visual_quat, visual_translation, rmse = self.camera_state_handler.get_visual_pose(
                measured_points=measured_points[measure_idx],
                reference_points=reference_points,
                quat=camera_quat)

            if not success:
                continue
            else:
                if self.smooth_vis_translation is None:
                    self.smooth_vis_translation = visual_translation
                    self.smooth_vis_quat = visual_quat
                else:
                    self.smooth_vis_translation = (self.vis_ema_param * visual_translation) + \
                                                  ((1 - self.vis_ema_param) * self.smooth_vis_translation)

                    self.smooth_vis_quat = quaternion.slerp(self.smooth_vis_quat,
                                                            visual_quat,
                                                            0, 1,
                                                            self.vis_ema_param).normalized()

                position_delta = self.smooth_vis_translation - camera_pos
                if np.linalg.norm(position_delta) > 0.05:
                    print(
                        f'Jump from {camera_pos} to {visual_translation} detected')
                    continue

                projected_z = project_to_tangent_space(self.smooth_vis_quat, self.smooth_vis_translation,
                                                       camera_quat, camera_pos)

                self.point_handler.append_points(measured_points_wf, reference_idx)

                quality_factor = (rmse / 0.0005) ** 2
                ukf_R = self.ukf.R.copy() * max(1.0, quality_factor)

                x_backup = self.ukf.x.copy()
                P_backup = self.ukf.P.copy()

                self.ukf.update(
                    z=projected_z,
                    R=ukf_R,
                    nominal_state=self.camera_state_handler.nominal_state.copy())

                if self.ukf.mahalanobis > self.GATE_THRESHOLD:
                    print(f"REJECTED: Outlier detected (Score: {self.ukf.mahalanobis:.2f})")

                    self.ukf.x = x_backup
                    self.ukf.P = P_backup

                    continue

                nominal_state = compose_fn(self.camera_state_handler.nominal_state.copy(),
                                           self.ukf.x.copy())

                self.camera_state_handler.set_state_from_nominal(nominal_state.copy())
                self.ukf.x.fill(0)

                self.logger.log_state(timestamp=vision_time, event='vision', state_vector=nominal_state)
                self.logger.log_vision(timestamp=vision_time, quat=visual_quat, translation=visual_translation)

                end_update = time.perf_counter()

                self.update_time = (end_update - start_update)

        pbar.close()

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
