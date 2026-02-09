import cv2
import numpy as np
import jax.numpy as jnp

from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment

import pyrealsense2

from helicopter.utils import PointQueue
from .point_detector import PointDetector


class PointHandler:
    def __init__(self,
                 detector: PointDetector,
                 queue_len: int = 50,
                 max_detections: int = 20) -> None:
        self.detector = detector
        self.maxlen = queue_len
        self.max_detections = max_detections

        self.point_map = np.empty((0, 3))
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

    def correct_points(self, points: np.ndarray, camera_position: np.ndarray, camera_quat: Rotation) -> np.ndarray:
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

        if len(marker_coords) <= 3:
            print('Not enough points')
            return None

        return marker_coords, keypoints

    def get_filter_inputs(self, measured_points, measure_idx, reference_points, reference_idx):
        m_slice = measured_points[measure_idx].astype(np.float32)
        r_slice = reference_points[reference_idx].astype(np.float32)

        n_valid = m_slice.shape[0]
        dim = m_slice.shape[1] if n_valid > 0 else 3

        if n_valid > self.max_detections:
            m_slice = m_slice[:self.max_detections]
            r_slice = r_slice[:self.max_detections]
            n_valid = self.max_detections

        z_padded = np.zeros((self.max_detections, dim), dtype=m_slice.dtype)
        ref_padded = np.zeros((self.max_detections, dim), dtype=r_slice.dtype)

        if n_valid > 0:
            z_padded[:n_valid] = m_slice
            ref_padded[:n_valid] = r_slice

        return (jnp.array(z_padded, dtype=jnp.float32),
                jnp.array(ref_padded, dtype=jnp.float32),
                jnp.array(n_valid, dtype=jnp.int32))

    def append_points(self, points: np.ndarray, point_idxs: list[int]):
        for idx, point in zip(point_idxs, points):
            self.points[idx].enqueue(point)
