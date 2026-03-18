from collections.abc import MutableMapping
import time
from multiprocessing.synchronize import Lock, Event
from multiprocessing.shared_memory import SharedMemory

import numpy as np
from scipy.spatial.transform import Rotation

import cv2
import pyrealsense2 as rs

from helicopter.configuration import HydraConfigurable
from helicopter.aircraft import Aircraft, FlightStates
from helicopter.utils import HelicopterModel

from .base import Visualizer
from .aruco_registry import ARUCOMarkerModel, aruco_registry


CAM_TO_BODY_MATRIX = np.array([
    [0., -1., 0.],
    [0., 0., -1.],
    [1., 0., 0.]
])
coordinate_transform = Rotation.from_matrix(CAM_TO_BODY_MATRIX)


@HydraConfigurable
class FlightVisualizer(Visualizer):
    def __init__(self, aircraft: Aircraft,
                 kill_signal: Event,
                 fps: float = 30.0):
        super().__init__()
        self.aircraft = aircraft

        self.server.initial_camera.position = (-0.25, -0.5, 0.1)
        self.server.initial_camera.look_at = (0.0, 0.0, 0.0)

        self.server.scene.add_grid(
            "/grid",
            width=5.0,
            height=5.0,
            position=np.array([0.0, 0.0, -0.15]),
            cell_size=0.1,
            cell_color=(0, 255, 0),
            cell_thickness=0.5,
            section_size=0.40,
            section_thickness=1.0,
            section_color=(0, 255, 0)
        )

        self.land_button = self.server.gui.add_button('Kill Flight')
        self.land_button.on_click(lambda _: self.kill_flight())

        helicopter_mesh = HelicopterModel().mesh()
        self.helicopter_handle = self.add_mesh(helicopter_mesh, '/camera')

        self.point_idxs = []

        self.last_position = np.array([0.0, 0.0, 0.0])

        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        parameters = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

        self.models = {}
        self.marker_size = 0.04

        self.camera_quat = None

        self.path_counter = 0
        self.fps = fps
        self.last_update_time = None

        self.is_running = False
        self.kill_signal = kill_signal

    def kill_flight(self):
        self.kill_signal.set()

    @staticmethod
    def unpack_intrinsics(intrinsics_dict: MutableMapping) -> rs.intrinsics:
        intrinsics = rs.intrinsics()

        intrinsics.width = intrinsics_dict['width']
        intrinsics.height = intrinsics_dict['height']
        intrinsics.ppx = intrinsics_dict['ppx']
        intrinsics.ppy = intrinsics_dict['ppy']
        intrinsics.fx = intrinsics_dict['fx']
        intrinsics.fy = intrinsics_dict['fy']
        intrinsics.model = rs.distortion(intrinsics_dict['model'])
        intrinsics.coeffs = intrinsics_dict['coeffs']

        return intrinsics

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
    def get_marker_depth(marker_corners, _id, depth_array, intrinsics, window_size=5):
        center_x = int(np.mean(marker_corners[:, 0]))
        center_y = int(np.mean(marker_corners[:, 1]))

        height, width = depth_array.shape

        half_w = window_size // 2
        y_min, y_max = max(0, center_y - half_w), min(height, center_y + half_w + 1)
        x_min, x_max = max(0, center_x - half_w), min(width, center_x + half_w + 1)

        roi = depth_array[y_min:y_max, x_min:x_max]

        valid_depths = roi[roi > 0]

        if len(valid_depths) == 0:
            raise RuntimeError(f"No valid depth points found for marker id {_id}")

        depth = np.median(valid_depths)
        point = rs.rs2_deproject_pixel_to_point(intrinsics,
                                                pixel=[np.mean(marker_corners[:, 0]), np.mean(marker_corners[:, 1])],
                                                depth=depth)
        return np.array([point[2], -point[0], -point[1]])

    @staticmethod
    def get_marker_quaternion(marker_corners, marker_size_meters, camera_matrix, dist_coeffs) -> tuple[bool, Rotation, np.ndarray]:
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

    def initialize(self, image_sm: SharedMemory, img_shape: list[int], intrinsics_dict: MutableMapping,
                   camera_quat_sm: SharedMemory, lock: Lock):
        local_intrinsics_dict = dict(intrinsics_dict)
        intrinsics = self.unpack_intrinsics(local_intrinsics_dict)

        ir_frame = np.empty(img_shape, dtype=np.int8)
        ir_buffer = np.ndarray((4,), dtype=np.int8, buffer=image_sm.buf)
        with lock:
            np.copyto(ir_frame, ir_buffer)

        if self.camera_quat is None:
            camera_quat = np.empty((4,), dtype=np.float64)
            camera_quat_buffer = np.ndarray((4,), dtype=np.float64, buffer=camera_quat_sm.buf)
            with lock:
                np.copyto(camera_quat, camera_quat_buffer)

            self.camera_quat = Rotation.from_quat(camera_quat)

        camera_matrix, dist_coeffs = self.realsense_to_opencv_intrinsics(intrinsics)
        corners, ids, rejected = self.detector.detectMarkers(ir_frame)
        for marker_id, marker_corners in zip(ids, corners):
            if marker_id not in aruco_registry.keys():
                print(f'Warning: ARUCO marker with id {marker_id} not registered. Skipping.')
                continue
            mesh_obj: ARUCOMarkerModel = aruco_registry[marker_id]
            mesh = mesh_obj.mesh()

            success, marker_rotation, marker_position = self.get_marker_quaternion(marker_corners,
                                                                               self.marker_size,
                                                                               camera_matrix,
                                                                               dist_coeffs)
            if not success:
                print(f"Warning: Failed to solve PnP for marker ID {marker_id}. Skipping.")
                continue

            corrected_marker_point = self.camera_quat.inv().apply(marker_position)
            total_rotation = self.camera_quat * marker_rotation
            mesh_handle = self.add_mesh(mesh, f'/aruco_mesh/{marker_id}',
                                        position=(corrected_marker_point - mesh_obj.marker_offset),
                                        orientation=total_rotation.as_quat(canonical=True))
            self.models[marker_id] = mesh_handle

        print("Aruco markers initialized")


    def update_helicopter(self, quat: Rotation, translation: np.ndarray):
        total_rotation = self.camera_quat * quat
        self.helicopter_handle.wxyz = total_rotation.as_quat(canonical=True, scalar_first=True)
        self.helicopter_handle.position = translation

        if np.linalg.norm(translation - self.last_position) > 0.005:
            line = np.vstack([self.last_position, translation])
            self.last_position = translation
            self.server.scene.add_line_segments(
                f"/line_segments/{self.path_counter}",
                points=np.expand_dims(line, 0),
                colors=(255, 255, 255),
                line_width=2.0,
            )
            self.path_counter += 1

    def loop(self):
        self.is_running = True
        while self.is_running:
            if self.kill_signal.is_set():
                raise RuntimeError('Visualizer detected kill signal. Shutting down.')

            if self.last_update_time is None:
                self.last_update_time = time.time()

            current_time = time.time()
            elapsed_time = current_time - self.last_update_time
            if elapsed_time < (1 / self.fps):
                time.sleep(0.001)
                continue
            else:
                self.last_update_time = current_time
                quat = self.aircraft.quaternion
                translation = self.aircraft.position

                render_quat = self.camera_quat.inverse() * quat
                self.update_helicopter(render_quat, translation)

    def cleanup(self):
        self.is_running = False
        self.server.stop()
