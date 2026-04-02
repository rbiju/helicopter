from abc import ABC, abstractmethod
import math
from pathlib import Path

import numpy as np
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation


class Kabsch:
    @staticmethod
    def kabsch(q: np.ndarray, p: np.ndarray):
        q_c = q - q.mean(axis=0)
        p_c = p - p.mean(axis=0)

        covar = p_c.T @ q_c
        u, _, vh = np.linalg.svd(covar)

        rotation = vh.T @ u.T

        if np.linalg.det(rotation) < 0:
            I = np.eye(3)
            I[2, 2] = -1.0
            rotation = vh.T @ I @ u.T

        translation = q.mean(axis=0) - rotation @ p.mean(axis=0)

        return Rotation.from_matrix(rotation), translation

    @staticmethod
    def apply(quat: Rotation, translation: np.ndarray, points: np.ndarray):
        return quat.apply(points) + translation


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
    def __init__(self, n: int,
                 k: int = 10,
                 inlier_threshold: float = 0.003,
                 reference_points_path: str = 'assets/point_clouds/green_syma.npy'):
        """

        Args:
            n: Number of best matching triangles to consider
            k: Top k matches to retrieve from KD Tree
            inlier_threshold: Distance within which point is considered an inlier post-registration
            reference_points_path: Reference point cloud, all rotations are relative to this initial orientation
        """
        super().__init__(reference_points_path)
        self.inlier_threshold = inlier_threshold
        lookup, data = self.compute_triangle_lookup_reference(self.reference_points, self.reference_distance_matrix)

        self.lookup = lookup
        self.kd_tree = KDTree(data, leafsize=2, balanced_tree=True)

        self.n = n
        self.k = k

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
        d, i = self.kd_tree.query(triangles, k=self.k)

        if len(d.shape) == 1:
            d = d.reshape(-1, 1)
            i = i.reshape(-1, 1)

        flat_d = d.ravel()
        flat_i = i.ravel()

        sample_indices = np.repeat(np.arange(len(triangles)), d.shape[1])

        top_n_idxs = np.argsort(flat_d)[:min(self.n, len(flat_d))]

        correspondences = []
        for idx in top_n_idxs:
            sample_idx = sample_indices[idx]
            ref_idx = flat_i[idx]

            sample_point_idxs = lookup_sample[int(sample_idx)]
            reference_point_idxs = lookup_ref[int(ref_idx)]

            correspondences.append({'reference': reference_point_idxs, 'sample': sample_point_idxs})

        return correspondences

    def get_alignment(self, sample_points: np.ndarray) -> tuple[Rotation, np.ndarray]:
        if len(sample_points) < 3:
            raise RuntimeError('Need at least 3 points to align')

        triangles, samples_lookup = self.compute_triangle_lookup_sample(sample_points)
        correspondences = self.get_top_n_triangle_correspondences(triangles, self.lookup, samples_lookup)

        best_inlier_count = 0
        best_error = np.inf
        best_alignment = (Rotation(np.array([0, 0, 0, 1.0])), np.array([0, 0, 0.0]))
        for correspondence in correspondences:
            ref_subset = self.reference_points[correspondence['reference']]
            sample_subset = sample_points[correspondence['sample']]
            rotation, translation = self.kabsch.kabsch(sample_subset, ref_subset)

            transformed_reference = self.kabsch.apply(rotation, translation, self.reference_points)
            distances = self.get_distance_matrix(transformed_reference, sample_points).min(axis=0)

            inlier_count = np.sum(distances < self.inlier_threshold)

            if inlier_count > best_inlier_count:
                best_inlier_count = inlier_count
                best_alignment = rotation, translation
                best_error = np.sum(distances)
            elif inlier_count == best_inlier_count:
                error = np.sum(distances)
                if error < best_error:
                    best_alignment = rotation, translation
                    best_error = error

        return best_alignment[0], best_alignment[1]
