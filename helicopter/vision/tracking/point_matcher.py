from abc import ABC, abstractmethod
import functools
import itertools
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.spatial.transform import Rotation


@jax.jit
def jax_get_distance_matrix(points_1, points_2):
    diff = points_1[:, None, :] - points_2[None, :, :]
    return jnp.linalg.norm(diff, axis=-1)


@jax.jit
def jax_single_kabsch(q, p):
    q_c = q - jnp.mean(q, axis=0)
    p_c = p - jnp.mean(p, axis=0)
    covar = p_c.T @ q_c
    u, _, vh = jnp.linalg.svd(covar, full_matrices=False)
    rotation = vh.T @ u.T
    det = jnp.linalg.det(rotation)
    rotation = jax.lax.cond(
        det < 0,
        lambda r: vh.T @ jnp.diag(jnp.array([1.0, 1.0, -1.0])) @ u.T,
        lambda r: r,
        rotation
    )
    translation = jnp.mean(q, axis=0) - rotation @ jnp.mean(p, axis=0)
    return rotation, translation


jax_batched_kabsch = jax.vmap(jax_single_kabsch, in_axes=(0, 0))


@jax.jit
def jax_apply_transform(rotation, translation, points):
    return points @ rotation.T + translation


@functools.partial(jax.jit, static_argnames=['k', 'n'])
def jax_get_top_n_correspondences(sample_features, ref_features, k, n):
    dists = jax_get_distance_matrix(sample_features, ref_features)
    vals, indices = jax.lax.top_k(-dists, k)
    flat_distances = -vals.ravel()
    flat_ref_indices = indices.ravel()
    flat_sample_indices = jnp.repeat(jnp.arange(dists.shape[0]), k)

    sort_idxs = jnp.argsort(flat_distances)
    top_n_idxs = sort_idxs[:n]

    return flat_sample_indices[top_n_idxs], flat_ref_indices[top_n_idxs]


@jax.jit
def jax_evaluate_alignments(rotations, translations, sample_points, ref_points, inlier_threshold):
    def eval_single(rot, trans):
        transformed_ref = jax_apply_transform(rot, trans, ref_points)
        distances = jax_get_distance_matrix(transformed_ref, sample_points).min(axis=0)
        inlier_count = jnp.sum(distances < inlier_threshold)
        error = jnp.sum(distances)
        return inlier_count, error

    inlier_counts, errors = jax.vmap(eval_single)(rotations, translations)
    max_inliers = jnp.max(inlier_counts)
    valid_mask = inlier_counts == max_inliers
    masked_errors = jnp.where(valid_mask, errors, jnp.inf)
    best_idx = jnp.argmin(masked_errors)

    return rotations[best_idx], translations[best_idx]


@jax.jit
def jax_get_triangle_features(distance_matrix, indices):
    d_ij = distance_matrix[indices[:, 0], indices[:, 1]]
    d_jk = distance_matrix[indices[:, 1], indices[:, 2]]
    d_ik = distance_matrix[indices[:, 0], indices[:, 2]]
    return jnp.stack([d_ij, d_jk, d_ik], axis=-1)


class Kabsch:
    @staticmethod
    def kabsch(q: np.ndarray, p: np.ndarray):
        rot, trans = jax_single_kabsch(jnp.array(q), jnp.array(p))
        return Rotation.from_matrix(np.array(rot)), np.array(trans)

    @staticmethod
    def apply(quat: Rotation, translation: np.ndarray, points: np.ndarray):
        return quat.apply(points) + translation


class PointMatcher(ABC):
    def __init__(self, reference_points_path: str = 'assets/point_clouds/green_syma.npy'):
        reference_points_path = Path(__file__).parents[3] / reference_points_path
        self.reference_points = jnp.array(np.load(str(reference_points_path)))
        self.reference_distance_matrix = jax_get_distance_matrix(self.reference_points, self.reference_points)
        self.sorted_reference_distance_matrix = jnp.sort(self.reference_distance_matrix, axis=-1)
        self.kabsch = Kabsch()

    @staticmethod
    def get_distance_matrix(points_1, points_2=None):
        if points_2 is None:
            points_2 = points_1
        return np.array(jax_get_distance_matrix(jnp.array(points_1), jnp.array(points_2)))

    @abstractmethod
    def get_alignment(self, sample_points: np.ndarray) -> tuple[Rotation, np.ndarray]:
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
        self.n = n
        self.k = k

        N = self.reference_points.shape[0]
        idx_i, idx_j, idx_k = jnp.meshgrid(jnp.arange(N), jnp.arange(N), jnp.arange(N), indexing='ij')
        self.ref_triangle_indices = jnp.stack([idx_i.ravel(), idx_j.ravel(), idx_k.ravel()], axis=-1)
        self.ref_features = jax_get_triangle_features(self.reference_distance_matrix, self.ref_triangle_indices)

    def get_alignment(self, sample_points: np.ndarray) -> tuple[Rotation, np.ndarray]:
        if len(sample_points) < 3:
            raise RuntimeError('Need at least 3 points to align')

        sample_points_jnp = jnp.array(sample_points)
        sample_dist_matrix = jax_get_distance_matrix(sample_points_jnp, sample_points_jnp)

        sample_triangle_indices = jnp.array(list(itertools.combinations(range(len(sample_points)), 3)))
        sample_features = jax_get_triangle_features(sample_dist_matrix, sample_triangle_indices)

        n_matches = min(self.n, len(sample_features) * self.k)
        best_sample_idxs, best_ref_idxs = jax_get_top_n_correspondences(
            sample_features, self.ref_features, k=self.k, n=n_matches
        )

        matched_sample_triangles = sample_points_jnp[sample_triangle_indices[best_sample_idxs]]
        matched_ref_triangles = self.reference_points[self.ref_triangle_indices[best_ref_idxs]]

        rotations, translations = jax_batched_kabsch(matched_sample_triangles, matched_ref_triangles)

        best_rot, best_trans = jax_evaluate_alignments(
            rotations, translations, sample_points_jnp, self.reference_points, self.inlier_threshold
        )

        return Rotation.from_matrix(np.array(best_rot)), np.array(best_trans)
