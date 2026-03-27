from abc import ABC, abstractmethod
import math
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation

from .kabsch import Kabsch


class PointMatcher(ABC):
    def __init__(self, reference_points_path: str = 'assets/point_clouds/green_syma.npy'):
        reference_points_path = Path(__file__).parents[3] / reference_points_path
        self.reference_points = np.load(str(reference_points_path))

        self.reference_distance_matrix = self.get_distance_matrix(self.reference_points)
        self.sorted_reference_distance_matrix = np.sort(self.reference_distance_matrix, axis=-1)

        self.kabsch = Kabsch()

    @staticmethod
    def get_distance_matrix(points_1, points_2=None):
        if points_2 is None:
            points_2 = points_1
        diff = points_1[:, None, :] - points_2[None, :, :]
        diff = np.linalg.norm(diff, axis=-1)

        return diff

    @abstractmethod
    def get_alignment(self, sample_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class TrianglePointMatcher(PointMatcher):
    def __init__(self, n: int, reference_points_path: str = 'assets/points_clouds/green_syma.npy'):
        super().__init__(reference_points_path)
        lookup, data = self.compute_triangle_lookup_reference(self.reference_points, self.reference_distance_matrix)

        self.lookup = lookup
        self.kd_tree = KDTree(data, leafsize=2, balanced_tree=True)

        self.n = n

    @staticmethod
    def compute_triangle_lookup_reference(points, distance_matrix):
        lookup = {}
        data = np.empty((points.shape[0] ** 3, 3))

        data_idx = 0
        for i in range(points.shape[0]):
            for j in range(points.shape[0]):
                for k in range(points.shape[0]):
                    triangle = np.array([distance_matrix[i, j], distance_matrix[j, k], distance_matrix[i, k]])
                    data[data_idx] = triangle
                    lookup[data_idx] = [i, j, k]
                    data_idx += 1

        return lookup, data

    def compute_triangle_lookup_sample(self, points):
        distance_matrix = self.get_distance_matrix(points)

        triangles = np.empty((math.comb(len(points), 3), 3))
        idx_lookup = {}

        idx = 0
        for i in range(points.shape[0]):
            for j in range(i + 1, points.shape[0]):
                for k in range(j + 1, points.shape[0]):
                    triangles[idx] = np.array([distance_matrix[i, j], distance_matrix[j, k], distance_matrix[i, k]])
                    idx_lookup[idx] = [i, j, k]
                    idx += 1

        return triangles, idx_lookup

    def get_top_n_triangle_correspondences(self, triangles, lookup_ref, lookup_sample):
        d, i = self.kd_tree.query(triangles)

        top_n_idxs = np.argsort(d)[:min(self.n, len(d))]

        correspondences = []
        for idx in top_n_idxs:
            sample_point_idxs = lookup_sample[idx]
            reference_point_idxs = lookup_ref[i[idx]]
            correspondences.append({r: s for r, s in zip(reference_point_idxs, sample_point_idxs)})

        return correspondences

    def get_alignment(self, sample_points: np.ndarray) -> tuple[Rotation, np.ndarray]:
        triangles, samples_lookup = self.compute_triangle_lookup_sample(sample_points)
        correspondences = self.get_top_n_triangle_correspondences(triangles, self.lookup, samples_lookup)

        alignment_error = np.inf
        best_alignment = (Rotation(np.array([0, 0, 0, 1.0])), np.array([0, 0, 0.0]))
        for correspondence in correspondences:
            ref_subset = self.reference_points[list(correspondence.keys())]
            sample_subset = sample_points[list(correspondence.values())]
            rotation, translation = self.kabsch.kabsch(sample_subset, ref_subset)

            transformed_reference = self.kabsch.apply(rotation, translation, ref_subset)
            error = self.get_distance_matrix(transformed_reference, sample_points).min(axis=0).sum()
            if error < alignment_error:
                alignment_error = error
                best_alignment = rotation, translation

        return best_alignment[0], best_alignment[1]
