import time
import threading

import numpy as np
import quaternion
from filterpy.kalman import UnscentedKalmanFilter

from helicopter.vision.d435i import D435i
from .camera_state_handler import CameraStateHandler
from .point_handler import PointHandler
from .utils import PointQueue

from .filter_functions import compose_fn, project_to_tangent_space, propagate


class MeasurementTool:
    def __init__(self,
                 device: D435i,
                 point_handler: PointHandler,
                 camera_state_handler: CameraStateHandler,
                 ukf: UnscentedKalmanFilter):
        self.device = device
        self.point_handler = point_handler
        self.camera_state_handler = camera_state_handler
        self.ukf = ukf

        self.ukf_Q = ukf.Q.copy()

        self.is_running = False

        self.timer = 0.0

        self.state_lock = threading.Lock()
        self.imu_thread = threading.Thread(target=self.imu_loop, daemon=True)

        self.max_dt = 0.0

    def initialize_orientation(self):
        accel_queue = PointQueue(500, np.array([0, 0, 0.0]))
        gyro_queue = PointQueue(500, np.array([0, 0, 0.0]))

        print("Initializing sensor orientation. Do not move camera")
        counter = 0
        while counter < 600:
            imu_frames = self.device.imu_pipeline.wait_for_frames()
            accel_data, ts_accel, gyro_data, ts_gyro = self.device.process_imu_frames(imu_frames)
            if accel_data is not None:
                accel_queue.enqueue(accel_data)
                gyro_queue.enqueue(gyro_data)
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

        self.camera_state_handler.accelerometer_bias = (quat * quaternion.quaternion(0,
                                                                                     *v_B) * quat.inverse()).imag - self.camera_state_handler.g
        self.camera_state_handler.gyro_bias = gyro_queue.mean()
        self.camera_state_handler.quaternion = quat

        print("Orientation initialized")

    def imu_loop(self):
        while self.is_running:
            imu_frames = self.device.imu_pipeline.wait_for_frames()

            synced_imu = self.device.get_synced_imu(imu_frames)
            if synced_imu is None:
                continue

            accel, gyro, timestamp = synced_imu

            if self.device.last_imu_time == 0:
                continue

            dt = timestamp - self.device.last_imu_time
            if dt <= 0:
                continue
            if dt > self.max_dt:
                self.max_dt = dt

            scale_factor = dt / (1 / 200)

            with self.state_lock:
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

                self.camera_state_handler.set_state_from_nominal(propagated_nominal)
                self.ukf.x.fill(0)

    def loop(self):
        try:
            print("Starting measurement in...")
            time.sleep(0.5)
            print('3')
            time.sleep(0.5)
            print('2')
            time.sleep(0.5)
            print('1')

            self.is_running = True
            debug_flag = False

            self.imu_thread.start()
            self.timer = time.time()
            while self.is_running:
                elapsed_time = time.time() - self.timer
                if elapsed_time > 0.0:
                    debug_flag = False
                if elapsed_time > 8.0:
                    self.is_running = False

                frames = self.device.pipeline.wait_for_frames()
                with self.state_lock:
                    camera_pos = self.camera_state_handler.position.copy()
                    camera_quat = self.camera_state_handler.quaternion.copy()

                depth_image, ts_depth, ir_image, ts_ir = self.device.process_frames(frames)
                if ir_image is not None:
                    if debug_flag:
                        print('whoa video')
                    measured_points_cf = self.point_handler.get_measured_points(ir_frame=ir_image,
                                                                                depth_frame=depth_image,
                                                                                intrinsics=self.device.intrinsics)
                    if measured_points_cf is None:
                        continue
                    else:
                        measured_points_wf, idxs = self.point_handler.match_points(
                            measured_points_cf,
                            camera_position=camera_pos,
                            camera_quat=camera_quat)

                        self.point_handler.append_points(measured_points_wf, idxs)

                        reference_points = self.point_handler.get_reference_points(registered_idxs=idxs)

                        success, visual_quat, visual_translation = self.camera_state_handler.get_visual_pose(
                            measured_points=measured_points_cf,
                            reference_points=reference_points,
                            quat=camera_quat)

                        if not success:
                            continue
                        else:
                            # position_delta = visual_translation - camera_pos
                            # # if np.linalg.norm(position_delta) > 0.01:
                            # #     print(
                            # #         f'Jump from {camera_pos} to {visual_translation} detected')
                            # #     continue

                            projected_z = project_to_tangent_space(visual_quat, visual_translation,
                                                                   camera_quat, camera_pos)

                            with self.state_lock:
                                self.ukf.update(
                                    z=projected_z,
                                    nominal_state=self.camera_state_handler.nominal_state.copy())

                                nominal_state = compose_fn(self.camera_state_handler.nominal_state.copy(),
                                                           self.ukf.x.copy())

                                self.camera_state_handler.set_state_from_nominal(nominal_state.copy())
                                self.ukf.x.fill(0)

        finally:
            print("Cleaning up sensor")
            self.is_running = False
            self.device.stop()

    def measure(self):
        self.loop()
