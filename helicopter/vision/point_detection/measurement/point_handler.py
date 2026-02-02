import numpy as np
import pyrealsense2
import quaternion
from scipy.optimize import linear_sum_assignment

from helicopter.vision import PointQueue
from .point_detector import PointDetector


class PointHandler:
    def __init__(self,
                 detector: PointDetector,
                 queue_len: int = 50) -> None:
        self.detector = detector
        self.maxlen = queue_len

        self.point_map = np.empty((0, 3), dtype=np.float64)
        self.points: dict[int, PointQueue] = {}

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
        comp = norm < self.detector.marker_tolerance
        if np.any(comp):
            return int(np.argmin(norm))
        else:
            return None

    def correct_points(self, points: np.ndarray, camera_position: np.ndarray,
                       camera_quat: quaternion.quaternion) -> np.ndarray:
        corrected = quaternion.rotate_vectors(camera_quat, points) + camera_position

        return corrected

    def register_points(self, points: np.ndarray):
        for point in points:
            if self.point_map.shape[0] < 1:
                self.add_point(point)
            else:
                norm = np.linalg.norm(self.point_map - point, axis=1)
                comp = norm > self.detector.marker_tolerance
                if np.all(comp):
                    self.add_point(point)

    def match_points(self, points: np.ndarray, camera_position: np.ndarray, camera_quat: quaternion.quaternion):
        corrected_points = self.correct_points(points, camera_position, camera_quat)
        self.register_points(corrected_points)

        diff = corrected_points[:, np.newaxis, :] - self.point_map[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)

        row_idx, col_idx = linear_sum_assignment(dist_matrix)

        return corrected_points[row_idx], row_idx, col_idx

    def get_measured_points(self, ir_frame: np.ndarray, depth_frame: np.ndarray,
                            intrinsics: pyrealsense2.intrinsics,
                            shift=2):
        keypoints = self.detector.detect(ir_frame)
        marker_coords = self.detector.get_point_coords(keypoints, depth_frame, intrinsics, shift)

        if len(marker_coords) <= 3:
            print('Not enough points')
            return None

        return marker_coords, keypoints

    def append_points(self, points: np.ndarray, point_idxs: list[int]):
        for idx, point in zip(point_idxs, points):
            self.points[idx].enqueue(point)
