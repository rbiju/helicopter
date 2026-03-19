from pathlib import Path

import cv2
import numpy as np

from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment

import pyrealsense2

from helicopter.utils import PointQueue
from ..point_detection.point_detector import PointDetector


class MeasurementPointHandler:
    def __init__(self,
                 detector: PointDetector,
                 queue_len: int = 50,
                 save_dir: str = "../../../notebooks/points") -> None:
        self.detector = detector
        self.maxlen = queue_len
        self.save_dir = Path(save_dir)

        self.point_map = np.empty((0, 3))
        self.points: dict[int, PointQueue] = {}

        self._next_id = 0

    def __repr__(self):
        summary = {'Num_Points': self._next_id,
                   'Points': self.points_coords}

        return (f"PointHandler: \n"
                f"{summary}")

    @property
    def next_id(self):
        out = self._next_id
        self._next_id += 1
        return out

    def add_point(self, point: np.ndarray) -> None:
        self.point_map = np.vstack((self.point_map, point))

        pq = PointQueue(self.maxlen)
        pq.enqueue(point)
        self.points[self.next_id] = pq

    @property
    def points_coords(self):
        point_list = [pq.mean() for pq in self.points.values()]
        points = np.array(point_list)

        return points

    @staticmethod
    def correct_points(points: np.ndarray, camera_position: np.ndarray, camera_quat: Rotation) -> np.ndarray:
        corrected = camera_quat.apply(points) + camera_position

        return corrected

    def deduplicate(self, points: np.ndarray) -> np.ndarray:
        if len(points) > 0:
            kept_indices = []
            dists = np.linalg.norm(points[:, None] - points[None, :], axis=2)
            np.fill_diagonal(dists, np.inf)

            processed = np.zeros(len(points), dtype=bool)
            for i in range(len(points)):
                if processed[i]:
                    continue

                kept_indices.append(i)
                processed[i] = True

                neighbors = np.where(dists[i] < self.detector.marker_tolerance)[0]
                processed[neighbors] = True

            return points[kept_indices]
        else:
            return points

    def register_points(self, points: np.ndarray):
        deduplicated = self.deduplicate(points)
        final_points = []

        for point in deduplicated:
            final_points.append(point)

            if self.point_map.shape[0] < 1:
                self.add_point(point)
            else:
                norm = np.linalg.norm(self.point_map - point, axis=1)
                is_new = np.all(norm > self.detector.marker_tolerance)
                if is_new:
                    self.add_point(point)

        if len(final_points) == 0:
            return np.empty((0, 3))

        return np.vstack(final_points)

    def match_points(self, points: np.ndarray, camera_position: np.ndarray, camera_quat: Rotation):
        corrected_points = self.correct_points(points, camera_position, camera_quat)
        self.register_points(corrected_points)

        diff = corrected_points[:, np.newaxis, :] - self.point_map[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)

        row_idx, col_idx = linear_sum_assignment(dist_matrix)

        return corrected_points[row_idx], row_idx, col_idx

    def get_measured_points(self, ir_frame: np.ndarray, depth_frame: np.ndarray,
                            intrinsics: pyrealsense2.intrinsics) -> tuple[np.ndarray, list[cv2.KeyPoint]] | None:
        keypoints = self.detector.detect(ir_frame)
        marker_coords = self.detector.get_points_coords(depth_frame, keypoints, intrinsics)

        if len(marker_coords) <= 0:
            return None

        return marker_coords, keypoints

    def append_points(self, points: np.ndarray, point_idxs: list[int]):
        for idx, point in zip(point_idxs, points):
            self.points[idx].enqueue(point)

    def save(self):
        np.save(self.save_dir / 'measured_points.npy', self.points_coords)
        np.save(self.save_dir / 'point_map.npy', self.point_map)
