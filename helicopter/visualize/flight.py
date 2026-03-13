import numpy as np
from scipy.spatial.transform import Rotation

import cv2
import pyrealsense2 as rs

from helicopter.configuration import HydraConfigurable
from helicopter.aircraft import AircraftManager
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
    def __init__(self, aircraft: AircraftManager):
        super().__init__()
        self.aircraft_manager = aircraft

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
    def get_marker_quaternion(marker_corners, marker_size_meters, camera_matrix, dist_coeffs) -> Rotation | None:
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
            return None

        rvec_flat = rvec.flatten()

        rotation = Rotation.from_rotvec(rvec_flat)
        rotation = coordinate_transform * rotation

        return rotation

    def initialize(self, ir_frame: np.ndarray, depth_frame: np.ndarray, intrinsics: rs.intrinsics, camera_quat: np.ndarray):
        if self.camera_quat is None:
            self.camera_quat = Rotation.from_quat(camera_quat)

        camera_matrix, dist_coeffs = self.realsense_to_opencv_intrinsics(intrinsics)
        corners, ids, rejected = self.detector.detectMarkers(ir_frame)
        for marker_id, marker_corners in zip(ids, corners):
            if marker_id not in aruco_registry.keys():
                print(f'Warning: ARUCO marker with id {marker_id} not registered. Skipping.')
                continue
            marker_point = self.get_marker_depth(marker_corners, marker_id, depth_frame, intrinsics)
            corrected_marker_point = self.camera_quat.inv().apply(marker_point)
            mesh_obj: ARUCOMarkerModel = aruco_registry[marker_id]
            mesh = mesh_obj.mesh()

            marker_rotation = self.get_marker_quaternion(marker_corners, self.marker_size, camera_matrix, dist_coeffs)
            if marker_rotation is None:
                print(f"Warning: Failed to solve PnP for marker ID {marker_id}. Skipping.")
                continue

            total_rotation = self.camera_quat * marker_rotation
            mesh_handle = self.add_mesh(mesh, f'/aruco_mesh/{marker_id}',
                                        position=(corrected_marker_point - mesh_obj.marker_offset),
                                        orientation=total_rotation.as_quat(canonical=True))
            self.models[marker_id] = mesh_handle

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
