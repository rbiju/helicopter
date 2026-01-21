import numpy as np
import pyrealsense2
import quaternion
import cv2
from scipy.optimize import linear_sum_assignment

import pyrealsense2 as rs

from .utils import PointQueue


class PointHandler:
    def __init__(self,
                 marker_radius: float = 0.01,
                 maxlen: int = 50) -> None:
        self.marker_radius = marker_radius

        self.maxlen = maxlen

        self.point_map = np.empty((0, 3), dtype=np.float64)
        self.points: dict[int, PointQueue] = {}

        params = cv2.SimpleBlobDetector.Params()

        params.filterByArea = True
        params.minArea = 5
        params.maxArea = 30

        params.filterByColor = True
        params.blobColor = 255

        params.filterByInertia = True
        params.minInertiaRatio = 0.7

        params.filterByCircularity = True
        params.minCircularity = 0.75

        params.filterByConvexity = True
        params.minConvexity = 0.9

        params.thresholdStep = 15
        params.minThreshold = 40
        params.maxThreshold = 200

        self.detector = cv2.SimpleBlobDetector.create(params)
        self.tophat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(25, 25))

        self._next_id = 0

    @property
    def next_id(self):
        out = self._next_id
        self._next_id += 1
        return out

    def add_point(self, point: np.ndarray) -> None:
        self.point_map = np.vstack((self.point_map, point))
        self.points[self.next_id] = PointQueue(maxlen=self.maxlen, init_value=point)

    @property
    def points_coords(self):
        point_list = [pq.mean() for pq in self.points.values()]
        points = np.array(point_list)

        return points

    def get_registered_idx(self, point: np.ndarray) -> bool | int | None:
        if self.point_map.shape[0] < 1:
            return None

        norm = np.linalg.norm(self.point_map - point, axis=1)
        comp = norm < self.marker_radius
        if np.any(comp):
            return int(np.argmin(norm))
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

    def get_marker_coords(self, depth_frame, valid_mask, circle, intrinsics: rs.intrinsics,
                          distance_threshold: float = 0.5,
                          sigma_factor=0.5) -> np.ndarray | None:
        h, w = depth_frame.shape
        center, radius, shift = circle

        ix, iy = int(center[0]), int(center[1])
        margin = int(radius + 2)

        x0, x1 = max(0, ix - margin), min(w, ix + margin + 1)
        y0, y1 = max(0, iy - margin), min(h, iy + margin + 1)

        depth_roi = depth_frame[y0:y1, x0:x1]
        valid_roi = valid_mask[y0:y1, x0:x1]

        roi_h, roi_w = depth_roi.shape
        local_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)

        local_center = (ix - x0, iy - y0)

        c_sub, r_sub = self.draw_subpixel_circle(local_center, radius, shift)
        cv2.circle(local_mask, c_sub, r_sub, 1, -1, shift=shift)

        ksize = int(radius) | 1
        gaussian_roi = cv2.GaussianBlur(local_mask.astype(float), (ksize, ksize),
                                        radius * sigma_factor) * valid_roi

        g_sum = gaussian_roi.sum()
        if g_sum <= 0:
            return None

        depth = np.sum(depth_roi * (gaussian_roi / g_sum))

        if depth > distance_threshold:
            return None

        point = rs.rs2_deproject_pixel_to_point(intrinsics, pixel=[center[0], center[1]], depth=depth)
        return np.array([point[2], -point[0], -point[1]])

    def correct_points(self, points: np.ndarray, camera_position: np.ndarray,
                       camera_quat: quaternion.quaternion) -> np.ndarray:
        rotated = quaternion.rotate_vectors(camera_quat, points)
        translated = rotated + camera_position

        return translated

    def register_points(self, points: np.ndarray):
        for point in points:
            if self.point_map.shape[0] < 1:
                self.add_point(point)
            else:
                norm = np.linalg.norm(self.point_map - point, axis=1)
                comp = norm > self.marker_radius
                if np.all(comp):
                    self.add_point(point)

    def match_points(self, points: np.ndarray, camera_position: np.ndarray, camera_quat: quaternion.quaternion):
        corrected_points = self.correct_points(points, camera_position, camera_quat)
        self.register_points(corrected_points)

        diff = corrected_points[:, np.newaxis, :] - self.point_map[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)

        row_idx, col_idx = linear_sum_assignment(dist_matrix)

        return corrected_points, col_idx

    def get_measured_points(self, ir_frame: np.ndarray, depth_frame: np.ndarray, intrinsics: pyrealsense2.intrinsics,
                            shift=3):
        valid_mask = 0 < depth_frame

        tophat = cv2.morphologyEx(ir_frame, cv2.MORPH_TOPHAT, self.tophat_kernel)
        clahe = self.clahe.apply(tophat)

        keypoints = self.detector.detect(clahe)

        points = []
        for kp in keypoints:
            marker_point = self.get_marker_coords(depth_frame, valid_mask, (kp.pt, kp.size / 2, shift), intrinsics)

            if marker_point is None:
                continue

            points.append(marker_point)

        if len(points) >= 3:
            return np.array(points)
        else:
            return None

    def append_points(self, points: np.ndarray, point_idxs: list[int]):
        for idx, point in zip(point_idxs, points):
            self.points[idx].enqueue(point)

    def get_reference_points(self, registered_idxs: list[int]):
        return self.point_map[registered_idxs]
