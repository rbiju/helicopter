import queue
import threading
import time
from multiprocessing import Queue
from multiprocessing.synchronize import Lock, Event
from multiprocessing.shared_memory import SharedMemory

import cv2
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation
import pyrealsense2 as rs

from helicopter.configuration import HydraConfigurable
from helicopter.aircraft import Aircraft, FlightStates
from helicopter.vision import D435i, ErrorStateSquareRootUnscentedKalmanFilter, UKFFactory
from helicopter.utils import PointQueue, Profiler, CommandBufferConstants

from .point_handler import TrackingPointHandler
from .filter_functions import propagate, transition_fn, measurement_fn, compose_fn


CAM_TO_BODY_MATRIX = np.array([
    [0., -1., 0.],
    [0., 0., -1.],
    [1., 0., 0.]
])
coordinate_transform = Rotation.from_matrix(CAM_TO_BODY_MATRIX)


@HydraConfigurable
class Tracker:
    def __init__(self, aircraft_sm: SharedMemory,
                 point_handler: TrackingPointHandler,
                 camera: D435i,
                 ukf_factory: UKFFactory,
                 kill_signal: Event,
                 simulation_fps: float = 250.):
        self.aircraft_buffer = np.ndarray(shape=(Aircraft.N,),
                                          dtype=Aircraft.dtype,
                                          buffer=aircraft_sm.buf)
        self.aircraft = None

        self.point_handler = point_handler
        self.camera = camera
        self.ukf: ErrorStateSquareRootUnscentedKalmanFilter = ukf_factory.filter()

        self.vision_thread = threading.Thread(target=self.vision_loop, daemon=True)
        self.vision_queue = queue.Queue(maxsize=5)

        self.camera_quat: Rotation = Rotation.from_rotvec(np.array([0, 0, 0.0]))
        self.initialized = False
        self.is_running = False

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        self.marker_size = 0.0427

        self.profiler = Profiler()

        self.last_simulated_time = 0.0
        self.simulation_fps = simulation_fps

        self.first_frame = False
        self.first_frame_time = 0.0

        self.kill_signal = kill_signal

    @staticmethod
    def realsense_to_opencv_intrinsics(rs_intrinsics: rs.intrinsics):
        fx = rs_intrinsics.fx
        fy = rs_intrinsics.fy
        cx = rs_intrinsics.ppx
        cy = rs_intrinsics.ppy

        camera_matrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float64)

        dist_coeffs = np.array(rs_intrinsics.coeffs, dtype=np.float64)

        return camera_matrix, dist_coeffs

    @staticmethod
    def get_marker_position(marker_corners,
                            marker_size_meters,
                            camera_matrix,
                            dist_coeffs) -> tuple[bool, Rotation, np.ndarray]:
        half_size = marker_size_meters / 2.0
        obj_points = np.array([
            [-half_size, half_size, 0],
            [half_size, half_size, 0],
            [half_size, -half_size, 0],
            [-half_size, -half_size, 0]
        ], dtype=np.float32)

        img_points = marker_corners[0].astype(np.float32)

        success, rvec, tvec = cv2.solvePnP(
            obj_points,
            img_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )

        if not success:
            return False, Rotation.from_rotvec(np.array([0, 0, 0.0])), np.array([0, 0, 0.0])

        rvec_flat = rvec.flatten()

        rotation = Rotation.from_rotvec(rvec_flat)
        rotation = coordinate_transform * rotation

        return True, rotation, tvec

    @staticmethod
    def align_aruco_to_ir(rvec_rgb, tvec_rgb, extrinsics: rs.extrinsics):
        R_ext = np.array(extrinsics.rotation).reshape(3, 3)

        T_ext = np.array(extrinsics.translation).reshape(3, 1)

        rvec_rgb = np.array(rvec_rgb).reshape(3, 1)
        tvec_rgb = np.array(tvec_rgb).reshape(3, 1)

        R_aruco_rgb, _ = cv2.Rodrigues(rvec_rgb)
        R_aruco_ir = np.dot(R_ext, R_aruco_rgb)

        T_aruco_ir = np.dot(R_ext, tvec_rgb) + T_ext
        rvec_ir, _ = cv2.Rodrigues(R_aruco_ir)

        return Rotation.from_rotvec(rvec_ir.flatten()), T_aruco_ir.flatten()

    def vision_loop(self):
        while self.is_running:
            frames = self.camera.pipeline.wait_for_frames()

            video = self.camera.process_frames(frames)

            if video.ir_image is not None:
                try:
                    self.vision_queue.put((video.depth_ts, video.ir_image, video.depth_image), block=False)
                except queue.Full:
                    self.vision_queue.get_nowait()
                    self.vision_queue.put((video.depth_ts, video.ir_image, video.depth_image), block=False)


    def initialize(self,
                   aruco_queue: Queue,
                   aircraft_lock: Lock):
        self.camera.start()

        camera_orientation_iters = 1000
        counter = 0
        accel_queue = PointQueue(maxlen=camera_orientation_iters)
        while counter < camera_orientation_iters:
            imu_frame = self.camera.imu_pipeline.wait_for_frames()
            imu_data = self.camera.process_imu_frames(imu_frame)
            if imu_data is not None:
                accel, _, _, _ = imu_data
                accel_queue.enqueue(accel)
                counter += 1

        v_B = accel_queue.mean()
        v_B /= np.linalg.norm(v_B)
        v_W = np.array([0, 0, 1.0])

        axis = np.cross(v_B, v_W)
        axis_norm = np.linalg.norm(axis)

        angle = np.arctan2(axis_norm, np.dot(v_B, v_W))

        if axis_norm > 1e-6:
            rotvec = axis * (angle / axis_norm)
        else:
            rotvec = np.array([np.pi, 0, 0]) if v_B[2] < 0 else np.zeros(3)

        self.camera_quat = Rotation.from_rotvec(rotvec)
        print(f'Initial Camera Acceleration (Normalized): {v_B}')
        print(f'Camera quaternion initialized: {self.camera_quat.as_rotvec(degrees=True)}')

        orientation_iters = 200
        print("Initializing helicopter orientation. Do not move aircraft.")
        counter = 0
        first_frame = False
        aruco_image = None
        while counter < orientation_iters:
            counter += 1
            frames = self.camera.pipeline.wait_for_frames()
            video = self.camera.process_frames(frames)
            if not first_frame:
                aruco_image = video.color_image
                first_frame = True

            measure_out = self.point_handler.get_measured_points(ir_frame=video.ir_image,
                                                                 depth_frame=video.depth_image,
                                                                 intrinsics=self.camera.intrinsics)
            if measure_out is not None:
                marker_coords, keypoints = measure_out
                self.point_handler.register_points(marker_coords)
            else:
                continue

        print(f'{len(self.point_handler.init_points_coords)} initial points detected')

        self.profiler.start('Init_Matching')
        r, t = self.point_handler.matcher.get_alignment(self.point_handler.init_points_coords)
        self.profiler.end('Init_Matching')

        if self.aircraft is None:
            self.aircraft = Aircraft(buffer=self.aircraft_buffer, lock=aircraft_lock)

        self.aircraft.quaternion = self.camera_quat * r
        self.aircraft.position = self.camera_quat.apply(t)

        aruco_dict = {}
        camera_matrix, dist_coeffs = self.realsense_to_opencv_intrinsics(self.camera.intrinsics)
        corners, ids, rejected = self.detector.detectMarkers(aruco_image)
        if ids is not None:
            for marker_id, marker_corners in zip(ids, corners):
                marker_id_int = int(marker_id[0])
                success, marker_rotation, marker_position = self.get_marker_position(marker_corners,
                                                                                     self.marker_size,
                                                                                     camera_matrix,
                                                                                     dist_coeffs)
                if not success:
                    print(f"Warning: Failed to solve PnP for marker ID {marker_id_int}. Skipping.")
                    continue

                aligned_marker_rotation, aligned_marker_position = self.align_aruco_to_ir(marker_rotation,
                                                                                          marker_position,
                                                                                          self.camera.color_ir_extrinsics)

                corrected_marker_point = self.camera_quat.apply(aligned_marker_position)
                corrected_rotation = self.camera_quat * aligned_marker_rotation

                aruco_dict[marker_id_int] = {'position': corrected_marker_point, 'rotation': corrected_rotation}

        aruco_queue.put(aruco_dict)

        print(f'Initialization complete. Aircraft detected @: {self.aircraft.position} \n '
              f'with orientation {self.aircraft.quaternion.as_rotvec(degrees=True)}')

    # TODO: add timestamp to aircraft buffer
    def loop(self, command_sm: SharedMemory, lock: Lock):
        command_buffer = np.ndarray(shape=(CommandBufferConstants.N,),
                                    dtype=CommandBufferConstants.dtype,
                                    buffer=command_sm.buf)
        commands = np.empty_like(command_buffer)
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

                vision_time = vision_time - self.first_frame_time

            except queue.Empty:
                time.sleep(0.001)
                continue

            self.profiler.start('Simulation')
            step_size = 1 / self.simulation_fps
            while self.last_simulated_time < vision_time:
                self.last_simulated_time += step_size

                with lock:
                    np.copyto(commands, command_buffer)

                nominal_state = jnp.array(self.aircraft.get_state_vector())

                flight_state = self.aircraft.flight_state
                if flight_state == FlightStates.IDLE or flight_state == FlightStates.DONE:
                    ground = True
                else:
                    ground = False
                propagated_nominal = propagate(nominal_state, step_size, commands, ground)

                self.profiler.start("UKF_Predict")
                self.ukf = self.ukf.predict(transition_fn=transition_fn,
                                            dt=step_size,
                                            nominal_state=nominal_state,
                                            propagated_nominal=propagated_nominal,
                                            commands=commands)
                self.profiler.end("UKF_Predict")

                self.aircraft.set_state_vector(np.asarray(propagated_nominal))

            self.profiler.end('Simulation')

            self.profiler.start('Detection')
            measured_out = self.point_handler.get_measured_points(ir_frame=ir_frame,
                                                                  depth_frame=depth_frame,
                                                                  intrinsics=self.camera.intrinsics)
            self.profiler.end("Detection")

            if measured_out is None:
                current_nominal_state = jnp.array(self.aircraft.get_state_vector())

                nominal_state = np.asarray(compose_fn(current_nominal_state, self.ukf.x))
                self.ukf = self.ukf.reset()

                self.aircraft.set_state_vector(nominal_state)
                continue
            else:
                measured_points, keypoints = measured_out
                # noinspection PyTypeChecker
                measured_points: np.ndarray = self.camera_quat.apply(measured_points)

                q = self.aircraft.quaternion
                t = self.aircraft.position.copy()
                nominal_state = jnp.array(self.aircraft.get_state_vector())

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

                nominal_state = np.asarray(compose_fn(nominal_state,
                                                      self.ukf.x))

                self.ukf = self.ukf.reset()

                self.aircraft.set_state_vector(nominal_state)
                self.profiler.end('UKF_Update')

    def cleanup(self):
        self.is_running = False

        if self.vision_thread.is_alive():
            self.vision_thread.join(timeout=1.0)
        self.camera.stop()
