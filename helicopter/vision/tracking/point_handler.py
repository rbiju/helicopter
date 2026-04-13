import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation as jaxRotation

import pyrealsense2

from helicopter.utils import PointQueue
from ..point_detection.point_detector import PointDetector

from .point_matcher import TrianglePointMatcher
from .icp import ICP


class TrackingPointHandler:
    def __init__(self,
                 detector: PointDetector,
                 matcher: TrianglePointMatcher,
                 icp: ICP,
                 max_queue_size: int = 50) -> None:
        self.detector = detector
        self.matcher = matcher
        self.icp = icp

        self.init_points : dict[int, PointQueue] = {}
        self.maxlen = max_queue_size

        self._next_id : int = 0

    @property
    def next_id(self) -> int:
        tmp = self._next_id
        self._next_id += 1
        return tmp

    @property
    def init_points_coords(self):
        point_list = [pq.mean() for pq in self.init_points.values()]
        points = np.array(point_list)

        return points

    def initial_points(self) -> np.ndarray:
        raw_points = self.init_points_coords

        if len(raw_points) == 0:
            return np.array([])

        aggregated_clusters = [[raw_points[0]]]

        for point in raw_points[1:]:
            centroids = np.array([np.mean(cluster, axis=0) for cluster in aggregated_clusters])

            norm = np.linalg.norm(centroids[:, 1:] - point[1:], axis=1)
            closest_idx = np.argmin(norm)

            if norm[closest_idx] <= self.detector.marker_tolerance:
                aggregated_clusters[closest_idx].append(point)
            else:
                aggregated_clusters.append([point])

        final_points = np.array([np.mean(cluster, axis=0) for cluster in aggregated_clusters])

        return final_points

    def add_point(self, point: np.ndarray) -> None:
        pq = PointQueue(self.maxlen)
        pq.enqueue(point)
        self.init_points[self.next_id] = pq

    def register_points(self, points: np.ndarray):
        for point in points:
            if len(self.init_points) < 1:
                self.add_point(point)
            else:
                # Depth noise is isolated to first dim, so just check for Y and Z
                # I think this is fine if the tolerance is tight enough
                norm = np.linalg.norm(self.init_points_coords[:, 1:] - point[1:], axis=1)
                closest_idx = np.argmin(norm)
                if norm[closest_idx] > self.detector.marker_tolerance:
                    self.add_point(point)
                else:
                    keys = list(self.init_points.keys())
                    closest_key = keys[closest_idx]
                    self.init_points[closest_key].enqueue(point)

    def get_measured_points(self, ir_frame: np.ndarray, depth_frame: np.ndarray,
                            intrinsics: pyrealsense2.intrinsics) -> tuple[np.ndarray, list[cv2.KeyPoint]] | None:
        keypoints = self.detector.detect(ir_frame)
        marker_coords = self.detector.get_points_coords(depth_frame, keypoints, intrinsics)

        if len(marker_coords) <= 0:
            return None

        return marker_coords, keypoints

    def pad_points(self, points):
        max_size = len(self.matcher.reference_points)
        current_size = points.shape[0]

        pad_length = max_size - current_size

        pad_array = np.zeros((pad_length, 3), dtype=points.dtype)
        padded_points = np.vstack([points, pad_array])

        valid_input_mask = np.zeros(max_size, dtype=bool)
        valid_input_mask[:current_size] = True

        return padded_points, valid_input_mask

    def refine_alignment(self, r: Rotation,
                         t: np.ndarray,
                         measured_points: np.ndarray) -> tuple[Rotation, np.ndarray]:
        q_jax = jaxRotation.from_quat(jnp.array(r.as_quat(canonical=True)))
        padded, valid_input_mask = self.pad_points(measured_points)
        _, _, q_jax, t_jax = self.icp.iterate(q_jax,
                                              t.copy(),
                                              padded,
                                              self.matcher.reference_points,
                                              valid_input_mask)
        return Rotation.from_quat(np.asarray(q_jax.as_quat(canonical=True))), t.copy()

    def get_point_correspondence(self, q_init: Rotation,
                                 t_init: np.ndarray,
                                 measured_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q_jax = jaxRotation.from_quat(jnp.array(q_init.as_quat(canonical=True)))
        padded, valid_input_mask = self.pad_points(measured_points)
        min_idxs, valid_mask, _, _ = self.icp.iterate(q_jax,
                                                t_init.copy(),
                                                padded,
                                                self.matcher.reference_points,
                                                valid_input_mask)
        min_idxs = np.asarray(min_idxs)
        valid_mask = np.asarray(valid_mask)

        measure_idxs = np.arange(len(measured_points))[valid_mask]
        reference_idxs = min_idxs[valid_mask]

        return measure_idxs, reference_idxs
