import queue
import threading
import time
from multiprocessing import Queue
from multiprocessing.synchronize import Lock, Event
from multiprocessing.shared_memory import SharedMemory

import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation

from helicopter.configuration import HydraConfigurable
from helicopter.aircraft import Aircraft
from helicopter.vision import D435i, ErrorStateSquareRootUnscentedKalmanFilter, UKFFactory
from helicopter.vision.point_detection import MarkerDetector
from helicopter.utils import PointQueue, Profiler, CommandBufferConstants

from .point_handler import TrackingPointHandler
from .filter_functions import propagate, transition_fn, measurement_fn, compose_fn


@HydraConfigurable
class Tracker:
    def __init__(self, aircraft_sm: SharedMemory,
                 point_handler: TrackingPointHandler,
                 marker_detector: MarkerDetector,
                 camera: D435i,
                 ukf_factory: UKFFactory,
                 kill_signal: Event,
                 simulation_fps: float = 250.):
        self.aircraft_buffer = np.ndarray(shape=(Aircraft.N,),
                                          dtype=Aircraft.dtype,
                                          buffer=aircraft_sm.buf)
        self.aircraft = None

        self.point_handler = point_handler
        self.marker_detector = marker_detector
        self.camera = camera
        self.ukf: ErrorStateSquareRootUnscentedKalmanFilter = ukf_factory.filter()

        self.marker_detector.activate(rs_intrinsics=self.camera.intrinsics,
                                      rs_extrinsics=self.camera.color_ir_extrinsics)

        self.vision_thread = threading.Thread(target=self.vision_loop, daemon=True)
        self.vision_queue = queue.Queue(maxsize=5)

        # camera space is in raw camera axes
        # world space is gravity aligned, zero yaw with origin coincident with IR sensor
        # table space has center as origin, X-Y aligned with edges (short-long), Z aligned with gravity
        self.camera_quat: Rotation = Rotation.from_rotvec(np.array([0, 0, 0.0]))    # for going from camera space to world space
        self.origin_position: np.ndarray = np.array([0.0, 0.0, 0.0])    # location of center of table in world space
        self.origin_quat: Rotation = Rotation.from_rotvec(np.array([0, 0, 0.0]))    # full quat for camera --> table
        self.initialized = False
        self.is_running = False

        self.profiler = Profiler()

        self.last_simulated_time = 0.0
        self.simulation_fps = simulation_fps

        self.first_frame = False
        self.first_frame_time = 0.0

        self.kill_signal = kill_signal

    def get_origin_position(self, rotations: list[Rotation], lengths: list[np.ndarray]) -> np.ndarray:
        current_position = np.array([0.0, 0.0, 0.0])
        for rotation, length in zip(rotations, lengths):
            current_position += length
        pass

    def world_to_table_space(self, points: np.ndarray) -> np.ndarray:
        p_shifted = points - self.origin_position
        point_table = self.origin_quat.inv().apply(p_shifted)

        return point_table

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
                   marker_queue: Queue,
                   origin_queue: Queue,
                   aircraft_lock: Lock):
        self.camera.start()

        # ------------------------------------------------------------
        # |                           IMU                            |
        # ------------------------------------------------------------

        camera_orientation_iters = 1000
        counter = 0
        accel_queue = PointQueue(maxlen=camera_orientation_iters)
        print('\nInitializing camera orientation.')
        while counter < camera_orientation_iters:
            counter += 1
            imu_frame = self.camera.imu_pipeline.wait_for_frames()
            imu_data = self.camera.process_imu_frames(imu_frame)
            if imu_data is not None:
                accel, _, _, _ = imu_data
                accel_queue.enqueue(accel)

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

        # ------------------------------------------------------------
        # |                         POINTS                           |
        # ------------------------------------------------------------
        marker_iters = 400
        print("\nInitializing helicopter orientation. Do not move aircraft.")
        counter = 0
        while counter < marker_iters:
            counter += 1
            frames = self.camera.pipeline.wait_for_frames()
            video = self.camera.process_frames(frames)
            self.profiler.start("Init_Detection")
            measure_out = self.point_handler.get_measured_points(ir_frame=video.ir_image,
                                                                 depth_frame=video.depth_image,
                                                                 intrinsics=self.camera.intrinsics)
            self.profiler.end("Init_Detection")

            if measure_out is not None:
                marker_coords, keypoints = measure_out
                self.point_handler.register_points(marker_coords)
            else:
                continue

        print(f'{len(self.point_handler.init_points_coords)} initial points detected')


        # ------------------------------------------------------------
        # |                         MARKERS                          |
        # ------------------------------------------------------------
        marker_iters = 100
        print("\nInitializing marker orientations.")
        counter = 0
        marker_dict = {}
        while counter < marker_iters:
            counter += 1
            frames = self.camera.pipeline.wait_for_frames()
            video = self.camera.process_frames(frames)

            self.profiler.start('Marker_Detection')
            detections = self.marker_detector.detect_markers(video.color_image)
            self.profiler.end('Marker_Detection')
            if len(detections) > 0:
                for detection in detections:
                    marker_dict[detection.id] = {'rotation': detection.rotation,
                                              'position': detection.position,}

        if not marker_dict:
            raise RuntimeError('Failed to find any markers.')

        print(f'Markers detected @: {marker_dict}')
        marker_queue.put(marker_dict)

        # ------------------------------------------------------------
        # |                         ORIGIN                           |
        # ------------------------------------------------------------

        try:
            origin = origin_queue.get(timeout=10.0)
        except queue.Full:
            raise RuntimeError('Render init led to timeout')

        world_space_marker_rotation = self.camera_quat * (marker_dict[origin['id']]['rotation'])
        yaw_only_marker_quat = Rotation.from_euler('Z',
                                                   world_space_marker_rotation.as_euler('ZYX')[0])
        self.origin_quat = (self.camera_quat *
                            yaw_only_marker_quat *
                            origin['rotation'])
        self.origin_position = (self.camera_quat.apply(marker_dict[origin['id']]['position'])
                                - self.origin_quat.apply(origin['position']))
        origin_queue.put({'origin_position': self.origin_position,
                          'origin_quat': self.origin_quat,
                          'camera_quat': self.camera_quat})

        # ------------------------------------------------------------
        # |                        AIRCRAFT                          |
        # ------------------------------------------------------------

        self.profiler.start('Init_Matching')
        world_space_coords = self.camera_quat.apply(self.point_handler.init_points_coords)
        r, t = self.point_handler.matcher.get_alignment(world_space_coords)
        self.profiler.end('Init_Matching')

        if self.aircraft is None:
            self.aircraft = Aircraft(buffer=self.aircraft_buffer, lock=aircraft_lock)

        self.aircraft.quaternion = self.origin_quat.inv() * r
        self.aircraft.position = self.world_to_table_space(t)

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

                ground = np.abs(nominal_state[6]) < 1e-2
                propagated_nominal = propagate(nominal_state, step_size, commands, ground)

                self.profiler.start("UKF_Predict")
                self.ukf = self.ukf.predict(transition_fn=transition_fn,
                                            dt=step_size,
                                            nominal_state=nominal_state,
                                            propagated_nominal=propagated_nominal,
                                            commands=commands)
                self.profiler.end("UKF_Predict")

                self.aircraft.timestamp = self.last_simulated_time
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
                world_measured_points = self.camera_quat.apply(measured_points)
                table_measured_points = self.world_to_table_space(world_measured_points)

                q = self.aircraft.quaternion
                t = self.aircraft.position.copy()
                nominal_state = jnp.array(self.aircraft.get_state_vector())

                self.profiler.start('Match_Points')
                measure_idx, ref_idx = self.point_handler.get_point_correspondence(q, t, table_measured_points)
                self.profiler.end('Match_Points')

                self.profiler.start('UKF_Update')
                z_points = table_measured_points[measure_idx]
                ref_points = self.point_handler.matcher.reference_points[ref_idx]

                for z_point, ref_point in zip(z_points, ref_points):
                    self.ukf = self.ukf.update(measurement_fn=measurement_fn,
                                               z_point=z_point,
                                               ref_point=ref_point,
                                               nominal_state=nominal_state)

                nominal_state = np.asarray(compose_fn(nominal_state,
                                                      self.ukf.x))

                self.ukf = self.ukf.reset()

                self.aircraft.timestamp = vision_time
                self.aircraft.set_state_vector(nominal_state)
                self.profiler.end('UKF_Update')

    def cleanup(self):
        self.is_running = False

        if self.vision_thread.is_alive():
            self.vision_thread.join(timeout=1.0)
        self.camera.stop()
