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

    @property
    def points_coords(self):
        point_list = [pq.mean() for pq in self.points.values()]
        points = np.array(point_list)

        return points

    def is_point_registered(self, point: np.ndarray) -> bool | int | None:
        if len(self.points) < 1:
            return None

        norm = np.linalg.norm(self.points_coords - point, axis=1)
        comp = norm < self.marker_radius
        if np.any(comp):
            return int(np.argmax(comp))
        else:
            return None

    def get_ellipse_coords(self, depth_frame, ellipse, intrinsics: rs.intrinsics, depth_scale: float, distance_threshold: float = 1.0,
                           sigma_factor=0.3) -> np.ndarray | None:
        (cx, cy), (MA, ma), angle = ellipse
        cx, cy = int(cx), int(cy)

        depth_frame = depth_frame * depth_scale
        valid_mask = depth_frame > 0

        mask = np.zeros(depth_frame.shape[:2], dtype=np.uint8)
        cv2.ellipse(mask, ellipse, 255, -1)

        ksize = int(max(MA, ma)) | 1
        sigma = ksize * sigma_factor
        gaussian_mask = cv2.GaussianBlur(mask.astype(float), (ksize, ksize), sigma) * valid_mask

        if gaussian_mask.sum() <= 0:
            return None

        gaussian_mask = gaussian_mask / gaussian_mask.sum()
        depth = np.sum(depth_frame * gaussian_mask)

        if depth > distance_threshold:
            return None

        point = rs.rs2_deproject_pixel_to_point(intrinsics, pixel=[cx, cy], depth=depth)
        return np.array([point[2], -point[0], -point[1]])

    def correct_point(self, point: np.ndarray, camera_position: np.ndarray,
                      camera_quat: quaternion.quaternion) -> np.ndarray:
        point = point - camera_position
        point = (camera_quat * quaternion.quaternion(0., *point) * camera_quat.inverse()).imag
        return point

    def get_measured_points(self, ir_frame: np.ndarray, depth_frame: np.ndarray, intrinsics: pyrealsense2.intrinsics,
                            depth_scale: float, camera_position: np.ndarray, camera_quat: quaternion.quaternion):
        edges = cv2.Canny(ir_frame, 500, 600)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        points = []
        registered_idxs = []
        for contour in contours:
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                (_, _), (MA, ma), angle = ellipse
                if cv2.contourArea(contour) > 0:
                    if np.abs(1. - ((np.pi * (MA / 2.) * (ma / 2.)) / cv2.contourArea(contour))) < 0.40:
                        ellipse_point = self.get_ellipse_coords(depth_frame, ellipse, intrinsics, depth_scale)
                        if ellipse_point is None:  # Invalid depth pixels or point too far away
                            continue
                        ellipse_point = self.correct_point(ellipse_point, camera_position, camera_quat)

                        registered_idx = self.is_point_registered(ellipse_point)

                        if registered_idx is None:
                            registered_idx = len(self.points)
                            self.points[registered_idx] = PointQueue(maxlen=self.maxlen, init_value=ellipse_point)

                        points.append(ellipse_point)
                        registered_idxs.append(registered_idx)

        if len(points) >= 3:
            return np.array(points), registered_idxs
        else:
            return None

    def get_reference_points(self, registered_idxs: list[int]):
        return self.points_coords[registered_idxs]
