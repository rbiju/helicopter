from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation

import cv2
from pupil_apriltags import Detector, Detection
import pyrealsense2 as rs


COB_MATRIX = np.array([
    [0, 0, 1],
    [-1, 0, 0],
    [0, -1, 0]
])


@dataclass
class DetectedMarker:
    id: int
    position: np.ndarray
    rotation: Rotation


class MarkerDetector(ABC):
    def __init__(self, marker_size_meters):
        self.rs_intrinsics = None
        self.intrinsic_matrix = None
        self.dist_coeffs = None
        self.extrinsics = None
        self.marker_size_meters = marker_size_meters

    def activate(self, rs_intrinsics: rs.intrinsics, rs_extrinsics: rs.extrinsics):
        self.rs_intrinsics = rs_intrinsics
        self.intrinsic_matrix, self.dist_coeffs = self.unpack_realsense_intrinsics(rs_intrinsics)
        self.extrinsics = rs_extrinsics

    @staticmethod
    def unpack_realsense_intrinsics(rs_intrinsics: rs.intrinsics):
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

    def align_rgb_to_ir(self, rotation_RGB: Rotation, position_RGB: np.ndarray):
        R_ext = Rotation.from_matrix(np.array(self.extrinsics.rotation).reshape(3, 3))
        t_ext = np.array(self.extrinsics.translation)

        rotation_IR = R_ext * rotation_RGB
        position_IR = R_ext.apply(position_RGB) + t_ext

        return position_IR, rotation_IR

    @abstractmethod
    def detect_markers(self, img) -> list[DetectedMarker]:
        raise NotImplementedError


class ARUCOMarkerDetector(MarkerDetector):
    def __init__(self, marker_size_meters: float = 0.0427,
                 parameters: Optional[cv2.aruco.DetectorParameters] = None):
        super().__init__(marker_size_meters)
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        if parameters is None:
            _parameters = cv2.aruco.DetectorParameters()
            _parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
            _parameters.cornerRefinementWinSize = 5
            _parameters.cornerRefinementMaxIterations = 100
            _parameters.cornerRefinementMinAccuracy = 0.001
            _parameters.minMarkerPerimeterRate = 0.01
            _parameters.minMarkerDistanceRate = 0.05
            _parameters.adaptiveThreshWinSizeMin = 3
            _parameters.adaptiveThreshWinSizeMax = 53
            _parameters.adaptiveThreshWinSizeStep = 5
            _parameters.adaptiveThreshConstant = 5
            _parameters.polygonalApproxAccuracyRate = 0.03
            _parameters.errorCorrectionRate = 0.6
        else:
            _parameters = parameters

        self.detector = cv2.aruco.ArucoDetector(aruco_dict, _parameters)

    def get_marker_position(self, marker_corners) -> tuple[bool, Rotation, np.ndarray]:
        half_size = self.marker_size_meters / 2.0
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
            self.intrinsic_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )

        if not success:
            return False, Rotation.from_rotvec(np.array([0, 0, 0.0])), np.array([0, 0, 0.0])

        R, _ = cv2.Rodrigues(rvec)

        flip_180 = np.array([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ], dtype=np.float64)

        R_offset = R @ flip_180

        rvec, _ = cv2.Rodrigues(R_offset)
        rvec = rvec.flatten()
        rvec = COB_MATRIX @ rvec

        translation = COB_MATRIX @ tvec.flatten()
        rotation = Rotation.from_rotvec(rvec)

        return True, rotation, translation

    def detect_markers(self, img) -> list[DetectedMarker]:
        corner_sets, ids, _ = self.detector.detectMarkers(img)
        detections = []
        for idx, corners in zip(ids, corner_sets):
            success, rotation_RGB, position_RGB = self.get_marker_position(marker_corners=corners)
            if success:
                position, rotation = self.align_rgb_to_ir(rotation_RGB=rotation_RGB,
                                                          position_RGB=position_RGB)
                detections.append(DetectedMarker(id=int(idx[0]),
                                                 position=position,
                                                 rotation=rotation,))
            else:
                continue

        return detections


class AprilTagMarkerDetector(MarkerDetector):
    def __init__(self, marker_size_meters: float = 0.0427,
                 families: str = "tag25h9"):
        super().__init__(marker_size_meters)
        self.detector = Detector(families=families)

    def detect_markers(self, img) -> list[DetectedMarker]:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        undistorted_img = cv2.undistort(gray_img, self.intrinsic_matrix, self.dist_coeffs)
        camera_params = (self.rs_intrinsics.fx, self.rs_intrinsics.fy, self.rs_intrinsics.ppx, self.rs_intrinsics.ppy)
        # noinspection PyTypeChecker
        detections: list[Detection] = self.detector.detect(
            undistorted_img,
            estimate_tag_pose=True,
            camera_params=camera_params,
            tag_size=self.marker_size_meters
        )

        results = []
        for tag in detections:
            position, rotation = self.align_rgb_to_ir(rotation_RGB=Rotation.from_matrix(tag.pose_R),
                                                      position_RGB=tag.pose_t)
            results.append(DetectedMarker(id=tag.tag_id,
                                          position=position,
                                          rotation=rotation))

        return results
