import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation
from functools import partial


class ICP:
    def __init__(self, distance_threshold: float = 5e-2, etol: float = 5e-3, max_iterations: int = 10):
        self.distance_threshold = distance_threshold
        self.etol = etol
        self.max_iterations = max_iterations

    @partial(jax.jit, static_argnums=(0,))
    def kabsch(self, P, Q, weights=None):
        if weights is None:
            weights = jnp.ones(P.shape[0])

        weights = jnp.expand_dims(weights, axis=-1)
        W_sum = jnp.sum(weights)

        centroid_P = jnp.sum(P * weights, axis=0) / jnp.maximum(W_sum, 1e-8)
        centroid_Q = jnp.sum(Q * weights, axis=0) / jnp.maximum(W_sum, 1e-8)

        P_centered = (P - centroid_P) * jnp.sqrt(weights)
        Q_centered = (Q - centroid_Q) * jnp.sqrt(weights)

        H = P_centered.T @ Q_centered
        U, _, Vt = jnp.linalg.svd(H)

        d = jnp.sign(jnp.linalg.det(Vt.T @ U.T))
        diag_matrix = jnp.diag(jnp.array([1.0, 1.0, d]))

        R = Vt.T @ diag_matrix @ U.T
        t = centroid_Q - R @ centroid_P

        return Rotation.from_matrix(R), t

    @partial(jax.jit, static_argnums=(0,))
    def apply(self, R: Rotation, t, points):
        return R.apply(points) + t

    @partial(jax.jit, static_argnums=(0,))
    def get_correspondence(self, reference_points, sample_points):
        diff = sample_points[:, None, :] - reference_points[None, :, :]
        distances = jnp.linalg.norm(diff, axis=-1)

        sample_to_ref_dist = jnp.min(distances, axis=1)
        sample_to_ref_idx = jnp.argmin(distances, axis=1)

        ref_to_sample_idx = jnp.argmin(distances, axis=0)

        sample_indices = jnp.arange(sample_points.shape[0])
        is_mutual = ref_to_sample_idx[sample_to_ref_idx] == sample_indices

        valid_mask = jnp.logical_and(
            sample_to_ref_dist < self.distance_threshold,
            is_mutual
        )

        return sample_to_ref_idx, valid_mask

    @partial(jax.jit, static_argnums=(0,))
    def iterate(self, q_old: Rotation, t_old, sample_points, reference_points, valid_input_mask):
        transformed_reference_points = self.apply(q_old, t_old, reference_points)

        def cond_fn(state):
            error, iter_count, q, t = state
            return jnp.logical_and(error > self.etol, iter_count < self.max_iterations)

        def body_fn(state):
            error, iter_count, q, t = state

            min_idxs, corr_mask = self.get_correspondence(transformed_reference_points, sample_points)
            combined_mask = jnp.logical_and(corr_mask, valid_input_mask)

            matched_reference = reference_points[min_idxs]

            safe_matched_ref = jnp.where(combined_mask[:, None], matched_reference, 0.0)
            safe_sample_pts = jnp.where(combined_mask[:, None], sample_points, 0.0)

            q_new, t_new = self.kabsch(safe_matched_ref, safe_sample_pts, weights=combined_mask)

            transformed_ref = self.apply(q_new, t_new, matched_reference)

            diff = jnp.linalg.norm(transformed_ref - safe_sample_pts, axis=-1)
            error_new = jnp.sum(jnp.where(combined_mask, diff, 0.0))

            return error_new, iter_count + 1, q_new, t_new

        init_state = (jnp.array(jnp.inf), jnp.int32(0), q_old, t_old)

        final_error, final_iter, q_final, t_final = jax.lax.while_loop(cond_fun=cond_fn,
                                                                       body_fun=body_fn,
                                                                       init_val=init_state)

        final_transformed_refs = self.apply(q_final, t_final, reference_points)
        final_min_idxs, final_corr_mask = self.get_correspondence(final_transformed_refs, sample_points)
        final_valid_mask = jnp.logical_and(final_corr_mask, valid_input_mask)

        return final_min_idxs, final_valid_mask, q_final, t_final
