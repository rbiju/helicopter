import jax
import jax.numpy as jnp
from jax.scipy.spatial.transform import Rotation
from functools import partial


class Kabsch:
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


class ICP:
    def __init__(self, distance_threshold: float = 1e-1, etol: float = 5e-3, max_iterations: int = 10):
        self.distance_threshold = distance_threshold
        self.etol = etol
        self.max_iterations = max_iterations
        self.kabsch = Kabsch()

    @partial(jax.jit, static_argnums=(0,))
    def get_correspondence(self, reference_points, sample_points):
        diff = sample_points[:, None, :] - reference_points[None, :, :]
        distances = jnp.linalg.norm(diff, axis=-1)

        min_distances = jnp.min(distances, axis=1)
        min_idxs = jnp.argmin(distances, axis=1)

        valid_mask = min_distances < self.distance_threshold

        return min_idxs, valid_mask

    @partial(jax.jit, static_argnums=(0,))
    def iterate(self, q_old: Rotation, t_old, sample_points, reference_points):
        transformed_reference_points = self.kabsch.apply(q_old, t_old, reference_points)

        def cond_fn(state):
            error, iter_count, q, t = state
            return jnp.logical_and(error > self.etol, iter_count < self.max_iterations)

        def body_fn(state):
            error, iter_count, q, t = state

            min_idxs, valid_mask = self.get_correspondence(transformed_reference_points, sample_points)
            matched_reference = transformed_reference_points[min_idxs]

            q_new, t_new = self.kabsch.kabsch(matched_reference, sample_points, weights=valid_mask)

            transformed_ref = self.kabsch.apply(q_new, t_new, matched_reference)

            diff = jnp.linalg.norm(transformed_ref - sample_points, axis=-1)
            error_new = jnp.sum(diff * valid_mask)

            return error_new, iter_count + 1, q_new, t_new

        init_state = (jnp.array(jnp.inf), jnp.int32(0), q_old, t_old)

        final_error, final_iter, q_final, t_final = jax.lax.while_loop(cond_fun=cond_fn,
                                                                       body_fun=body_fn,
                                                                       init_val=init_state)

        final_transformed_refs = q_final.apply(reference_points) + t_final
        final_min_idxs, final_valid_mask = self.get_correspondence(final_transformed_refs, sample_points)

        return final_min_idxs, final_valid_mask
