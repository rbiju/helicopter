import time

import numpy as np
import quaternion

from ultralytics import YOLO

from helicopter.vision.d435i import D435i
from helicopter.vision.point_detection import HelicopterYOLO, GPUImagePreprocessor
from helicopter.vision.point_detection.measurement import YOLOPointDetector
from helicopter.vision.measurement.logger import StateLogger
from helicopter.vision.measurement.scanner import CameraStateHandler, PointHandler, PointQueue


class VisionOnlyMeasurementTool:
    def __init__(self,
                 device: D435i,
                 point_handler: PointHandler,
                 camera_state_handler: CameraStateHandler,
                 logger: StateLogger):
        self.device = device
        self.point_handler = point_handler
        self.camera_state_handler = camera_state_handler
        self.logger = logger

        self.is_running = False

        self.timer = 0.0

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

        self.camera_state_handler.accelerometer_bias = ((quat * quaternion.quaternion(0,
                                                                                      *v_B) * quat.inverse()).imag -
                                                        self.camera_state_handler.g)
        self.camera_state_handler.gyro_bias = gyro_queue.mean()
        self.camera_state_handler.quaternion = quat

        print("Orientation initialized")

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

            measurement_count = 0
            self.timer = time.time()
            while self.is_running:
                elapsed_time = time.time() - self.timer
                if elapsed_time > 4.5:
                    debug_flag = True
                if elapsed_time > 5.0:
                    self.is_running = False

                frames = self.device.pipeline.wait_for_frames()

                camera_pos = self.camera_state_handler.position.copy()
                camera_quat = self.camera_state_handler.quaternion.copy()

                depth_image, ts_depth, ir_image, ts_ir, laser_state = self.device.process_frames(frames)
                if ir_image is not None:
                    if debug_flag:
                        print('whoa video')

                    start_detect = time.perf_counter()
                    measured_points_out = self.point_handler.get_measured_points(ir_frame=ir_image,
                                                                                 depth_frame=depth_image,
                                                                                 intrinsics=self.device.intrinsics)
                    if measured_points_out is None:
                        print("Not enough points")
                        continue
                    else:
                        measured_points_cf = measured_points_out
                        measured_points_wf, idxs = self.point_handler.match_points(
                            measured_points_cf,
                            camera_position=camera_pos,
                            camera_quat=camera_quat)

                        reference_points = self.point_handler.get_reference_points(registered_idxs=idxs)

                        start_pose = time.perf_counter()
                        success, visual_quat, visual_translation = self.camera_state_handler.get_visual_pose(
                            measured_points=measured_points_cf,
                            reference_points=reference_points,
                            quat=camera_quat)
                        end_pose = time.perf_counter()
                        pose_time = end_pose - start_pose

                        if not success:
                            print("Failed to get visual pose")
                            continue
                        else:
                            position_delta = visual_translation - camera_pos
                            if np.linalg.norm(position_delta) > 0.05:
                                print(
                                    f'Jump from {camera_pos} to {visual_translation} detected')
                                continue

                            self.point_handler.append_points(measured_points_wf, idxs)

                            self.camera_state_handler.quaternion = visual_quat
                            self.camera_state_handler.position = visual_translation

                            self.logger.log_state(timestamp=ts_ir, event='vision',
                                                  state_vector=self.camera_state_handler.nominal_state)

                            measurement_count += 1

                            end_detect = time.perf_counter()

                            detected_time = (end_detect - start_detect)

        finally:
            print("Cleaning up sensor")
            self.is_running = False
            self.device.stop()

    def measure(self):
        self.loop()


if __name__ == '__main__':
    _device = D435i(enable_motion=True, projector_power=360., autoexpose=False,
                    exposure_time=1600)
    _point_handler = PointHandler(
        detector=YOLOPointDetector(
            model=HelicopterYOLO(preprocessor=GPUImagePreprocessor(imgsz=_device.IR_RESOLUTION),
                                 model=YOLO('/home/ray/yolo_models/helicopter/measure/weights/best.engine',
                                            task='detect'),
                                 conf=0.75),
            marker_tolerance=0.01,
            marker_size=0.003,
            marker_size_tolerance=0.3,
            distance_threshold=0.5
        ),
        queue_len=75)

    tool = VisionOnlyMeasurementTool(
        device=_device,
        point_handler=_point_handler,
        camera_state_handler=CameraStateHandler(),
        logger=StateLogger(save_dir="../../../notebooks/vision_only.csv"), )
    tool.measure()
