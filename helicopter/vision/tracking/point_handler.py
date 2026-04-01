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
                 max_queue_size: int = 50,
                 max_icp_iters: int = 10) -> None:
        self.detector = detector
        self.matcher = matcher
        self.icp = ICP()

        self.init_points : dict[int, PointQueue] = {}
        self.maxlen = max_queue_size

        self._next_id : int = 0

        self.max_icp_iters = max_icp_iters

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

    def add_point(self, point: np.ndarray) -> None:
        pq = PointQueue(self.maxlen)
        pq.enqueue(point)
        self.init_points[self.next_id] = pq

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

        for point in deduplicated:
            if len(self.init_points) < 1:
                self.add_point(point)
            else:
                # Depth noise is isolated to first dim, so just check for Y and Z
                # I think this is fine since the helicopter is facing side-on during init
                norm = np.linalg.norm(self.init_points_coords[:, 1:] - point[1:], axis=1)
                closest_point = np.argmin(norm)
                if norm[closest_point] > self.detector.marker_tolerance:
                    self.add_point(point)
                else:
                    self.init_points[closest_point].enqueue(point)

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
        pad_array = np.full((pad_length, 3), np.inf, dtype=points.dtype)

        padded_points = np.vstack([points, pad_array])

        return padded_points

    def get_point_correspondence(self, q_init: Rotation,
                                 t_init: np.ndarray,
                                 measured_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q_jax = jaxRotation.from_quat(jnp.array(q_init.as_quat(canonical=True)))
        min_idxs, valid_mask, _, _ = self.icp.iterate(q_jax,
                                                t_init.copy(),
                                                self.pad_points(measured_points),
                                                self.matcher.reference_points)
        min_idxs = np.asarray(min_idxs)
        valid_mask = np.asarray(valid_mask)

        measure_idxs = np.arange(len(measured_points))[valid_mask]
        reference_idxs = min_idxs[valid_mask]

        return measure_idxs, reference_idxs
