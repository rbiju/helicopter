import numpy as np
import pyrealsense2
import quaternion
import cv2

import pyrealsense2 as rs

from .utils import PointQueue


class PointHandler:
    def __init__(self, ir_threshold: int = 215,
                 marker_radius: float = 0.005,
                 maxlen: int = 100) -> None:
        self.ir_threshold = ir_threshold
        self.marker_radius = marker_radius

        self.maxlen = maxlen

        self.points: dict[int, PointQueue] = {}

        params = cv2.SimpleBlobDetector.Params()

        params.filterByArea = True
        params.minArea = 7
        params.maxArea = 200

        params.filterByColor = True
        params.blobColor = 255

        params.filterByInertia = True
        params.minInertiaRatio = 0.6

        params.filterByCircularity = True
        params.minCircularity = 0.5

        params.filterByConvexity = True
        params.minConvexity = 0.9

        params.thresholdStep = 5
        params.minThreshold = 20
        params.maxThreshold = 255

        self.detector = cv2.SimpleBlobDetector.create(params)
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(20, 20))

    @property
    def points_coords(self):
        point_list = [pq.mean() for pq in self.points.values()]
        points = np.array(point_list)

        return points

    def get_registered_idx(self, point: np.ndarray) -> bool | int | None:
        if len(self.points) < 1:
            return None

        norm = np.linalg.norm(self.points_coords - point, axis=1)
        comp = norm < self.marker_radius
        if np.any(comp):
            return int(np.argmax(comp))
        else:
            return None

    @staticmethod
    def draw_subpixel_circle(center, radius, shift):
        factor = 1 << shift

        cx, cy = center

        cx_rounded = round(cx * factor)
        cy_rounded = round(cy * factor)
        radius_rounded = round(radius * factor)

        return (cx_rounded, cy_rounded), radius_rounded

    def get_marker_coords(self, depth_frame, valid_mask, circle, intrinsics: rs.intrinsics, distance_threshold: float = 1.0,
                          sigma_factor=0.3) -> np.ndarray | None:
        center, radius, shift = circle

        mask = np.zeros(depth_frame.shape[:2], dtype=np.uint8)
        cv2.circle(mask, center, radius, 1, -1, shift=shift)

        ksize = int(radius / (1 << shift)) | 1
        gaussian_mask = cv2.GaussianBlur(mask.astype(float), (ksize, ksize), radius * sigma_factor) * valid_mask

        if gaussian_mask.sum() <= 0:
            return None

        gaussian_mask = gaussian_mask / gaussian_mask.sum()
        depth = np.sum(depth_frame * gaussian_mask)

        if depth > distance_threshold:
            return None

        point = rs.rs2_deproject_pixel_to_point(intrinsics, pixel=[center[0], center[1]], depth=depth)
        return np.array([point[2], -point[0], -point[1]])

    def correct_point(self, point: np.ndarray, camera_position: np.ndarray,
                      camera_quat: quaternion.quaternion) -> np.ndarray:
        point = point - camera_position
        point = (camera_quat * quaternion.quaternion(0., *point) * camera_quat.inverse()).imag
        return point

    def get_measured_points(self, ir_frame: np.ndarray, depth_frame: np.ndarray, intrinsics: pyrealsense2.intrinsics,
                            camera_position: np.ndarray, camera_quat: quaternion.quaternion, shift=3):
        valid_mask = 0 < depth_frame

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        tophat = cv2.morphologyEx(ir_frame, cv2.MORPH_TOPHAT, kernel)
        clahe = self.clahe.apply(tophat)

        for _ in range(2):
            clahe = self.clahe.apply(clahe)

        keypoints = self.detector.detect(clahe)

        points = []
        registered_idxs = []
        for kp in keypoints:
            center, radius = self.draw_subpixel_circle(kp.pt, kp.size / 2, shift)
            marker_point = self.get_marker_coords(depth_frame, valid_mask, (center, radius, shift), intrinsics)

            if marker_point is None:
                continue

            marker_point = self.correct_point(marker_point, camera_position, camera_quat)

            registered_idx = self.get_registered_idx(marker_point)

            if registered_idx is None:
                registered_idx = len(self.points)
                self.points[registered_idx] = PointQueue(maxlen=self.maxlen, init_value=marker_point)

            points.append(marker_point)
            registered_idxs.append(registered_idx)

        if len(points) >= 3:
            return np.array(points), registered_idxs
        else:
            return None

    def get_reference_points(self, registered_idxs: list[int]):
        return self.points_coords[registered_idxs]
