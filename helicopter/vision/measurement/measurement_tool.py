import time

import numpy as np
import quaternion
from filterpy.kalman import UnscentedKalmanFilter

from helicopter.vision.d435i import D435i
from .camera_state_handler import CameraStateHandler
from .point_handler import PointHandler
from .utils import PointQueue

from .filter_functions import compose_fn, propagate


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

        self.is_running = False

        self.timer = 0.0

    async def on_press(self, key):
        if key == "q":
            self.is_running = False

    def initialize_orientation(self):
        accel_queue = PointQueue(500, np.array([0, 0, 1.0]))

        print("Initializing sensor orientation. Do not move camera")
        counter = 0
        while counter < 1000:
            accel_frames = self.device.imu_pipeline.wait_for_frames()
            accel_data, ts_accel, gyro_data, ts_gyro = self.device.process_imu_frames(accel_frames)
            if accel_data is not None:
                accel_queue.enqueue(accel_data)
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

        self.camera_state_handler.quaternion = quat
        self.camera_state_handler.g = v_B_norm * v_W_unit

        print("Orientation initialized")

    def loop(self):
        try:
            print("Starting measurement")
            self.is_running = True
            self.timer = time.time()

            debug_flag = False
            while self.is_running:
                imu_frames = self.device.imu_pipeline.poll_for_frames()
                if imu_frames is not None:

                    if debug_flag:
                        print('whoa imu')

                    synced_imu = self.device.get_synced_imu(imu_frames)
                    if synced_imu is None:
                        continue
                    else:
                        accel, gyro, timestamp = synced_imu
                        dt = timestamp - self.device.last_imu_time

                        propagated_nominal = propagate(self.camera_state_handler.nominal_state.copy(), dt, accel, gyro, self.camera_state_handler.g)

                        self.ukf.predict(dt=dt,
                                         nominal_state=self.camera_state_handler.nominal_state.copy(),
                                         propagated_nominal=propagated_nominal,
                                         accelerometer=accel,
                                         gyro=gyro,
                                         g_world=self.camera_state_handler.g)

                        self.camera_state_handler.set_state_from_nominal(propagated_nominal)

                        self.ukf.x.fill(0)

                frames = self.device.pipeline.poll_for_frames()
                depth_frame = frames.get_depth_frame()
                ir_frame = frames.get_infrared_frame()
                if depth_frame and ir_frame:
                    depth_image, ts_depth, ir_image, ts_ir = self.device.process_frames(frames)
                    if ir_image is not None:
                        if debug_flag:
                            print('whoa video')
                        nominal_state = compose_fn(self.camera_state_handler.nominal_state, self.ukf.x.copy())
                        measured_points_cf = self.point_handler.get_measured_points(ir_frame=ir_image,
                                                                                    depth_frame=depth_image,
                                                                                    intrinsics=self.device.intrinsics)
                        if measured_points_cf is None:
                            continue
                        else:
                            measured_points_wf, idxs = self.point_handler.match_points(measured_points_cf,
                                                                                       camera_position=nominal_state[
                                                                                           4:7],
                                                                                       camera_quat=quaternion.quaternion(
                                                                                           *nominal_state[0:4]))
                            self.point_handler.append_points(measured_points_wf, idxs)
                            reference_points = self.point_handler.get_reference_points(registered_idxs=idxs)

                            visual_quat, visual_translation = self.camera_state_handler.get_visual_pose(
                                measured_points=measured_points_cf,
                                reference_points=reference_points)

                            self.ukf.update(
                                z=np.concatenate((quaternion.as_float_array(visual_quat), visual_translation)),
                                nominal_state=self.camera_state_handler.nominal_state.copy(), )

                            nominal_state = compose_fn(self.camera_state_handler.nominal_state.copy(), self.ukf.x.copy())

                            self.camera_state_handler.set_state_from_nominal(nominal_state.copy())
                            self.ukf.x = np.zeros(15)
                if time.time() - self.timer > 2.0:
                    debug_flag = True
                if time.time() - self.timer > 3.0:
                    self.is_running = False

                time.sleep(1e-5)

        finally:
            print("Cleaning up sensor")
            self.is_running = False
            self.device.stop()

    def measure(self):
        self.loop()
