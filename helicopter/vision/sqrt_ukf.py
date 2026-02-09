import jax
import jax.numpy as jnp
from jax import lax
from jax.tree_util import register_pytree_node_class
from functools import partial


@register_pytree_node_class
class ErrorStateSquareRootUnscentedKalmanFilter:
    def __init__(self, x, S, Q, R, alpha=1e-3, beta=2.0, kappa=0.0):
        n_dim = len(x)
        self.n_dim = n_dim
        self.x = x
        self.S = S

        if not (type(Q) is object or Q is None or isinstance(Q, ErrorStateSquareRootUnscentedKalmanFilter)):
            Q = jnp.linalg.cholesky(jnp.array(Q)).T
        self.Q_upper = Q

        if not (type(R) is object or R is None or isinstance(R, ErrorStateSquareRootUnscentedKalmanFilter)):
            R = jnp.linalg.cholesky(jnp.array(R)).T
        self.R_upper = R

        lamb = alpha ** 2 * (n_dim + kappa) - n_dim
        self.c = n_dim + lamb

        Wm = jnp.full(2 * n_dim + 1, 0.5 / self.c)
        self.Wm = Wm.at[0].set(lamb / self.c)

        Wc = self.Wm.copy()
        self.Wc = Wc.at[0].add(1 - alpha ** 2 + beta)

    def tree_flatten(self):
        children = (self.x, self.S, self.Q_upper, self.R_upper, self.Wm, self.Wc)
        aux_data = {'n_dim': self.n_dim, 'c': self.c}
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)
        obj.n_dim = aux_data['n_dim']
        obj.c = aux_data['c']
        obj.x, obj.S, obj.Q_upper, obj.R_upper, obj.Wm, obj.Wc = children
        return obj

    def reset(self, x=None):
        new_x = x if x is not None else jnp.zeros(self.n_dim)

        obj = self.tree_unflatten(
            {'n_dim': self.n_dim, 'c': self.c},
            (new_x, self.S, self.Q_upper, self.R_upper, self.Wm, self.Wc)
        )
        return obj

    @staticmethod
    @jax.jit
    def _cholesky_downdate(S, v):
        n = v.shape[0]

        def body(i, carry):
            S_curr, v_curr = carry
            sq_diff = jnp.square(S_curr[i, i]) - jnp.square(v_curr[i])

            r = jnp.sqrt(jnp.maximum(sq_diff, 1e-12))

            safe_diag = jnp.where(S_curr[i, i] == 0, 1e-12, S_curr[i, i])
            c = r / safe_diag
            s = v_curr[i] / safe_diag

            row = (S_curr[i, :] - s * v_curr) / c

            v_next = c * v_curr - s * row

            mask = jnp.arange(n) >= i
            row_masked = jnp.where(mask, row, 0.0)
            S_next = S_curr.at[i, :].set(row_masked)

            return S_next, v_next

        S_final, _ = lax.fori_loop(0, n, body, (S, v))
        return S_final

    def _generate_sigma_points(self, x, S_upper):
        sig = jnp.sqrt(self.c) * S_upper.T
        points_plus = x + sig.T
        points_minus = x - sig.T
        return jnp.concatenate([x[None, :], points_plus, points_minus], axis=0)

    @partial(jax.jit, static_argnames=['transition_fn'])
    def predict(self, transition_fn, dt, **kwargs):
        points = self._generate_sigma_points(self.x, self.S)
        points_pred = jax.vmap(lambda p: transition_fn(p, dt, **kwargs))(points)

        x_pred = jnp.dot(self.Wm, points_pred)
        diff = points_pred - x_pred

        weight_sqrt = jnp.sqrt(self.Wc[1])
        weighted_diff = diff[1:].T * weight_sqrt

        M = jnp.hstack([weighted_diff, self.Q_upper.T])
        _, S_pred = lax.linalg.qr(M.T, full_matrices=False)

        diff_0 = diff[0]

        def _downdate(s):
            return self._cholesky_downdate(s, diff_0 * jnp.sqrt(-self.Wc[0]))

        def _update(s):
            return lax.linalg.cholesky_update(s, diff_0 * jnp.sqrt(self.Wc[0]))

        S_pred = lax.cond(self.Wc[0] < 0, _downdate, _update, S_pred)

        return self.tree_unflatten(
            {'n_dim': self.n_dim, 'c': self.c},
            (x_pred, S_pred, self.Q_upper, self.R_upper, self.Wm, self.Wc)
        )

    @partial(jax.jit, static_argnames=['measurement_fn'])
    def update(self, measurement_fn, z_point, **kwargs):
        points = self._generate_sigma_points(self.x, self.S)

        Z_sig = jax.vmap(lambda p: measurement_fn(p, **kwargs))(points)
        z_pred = jnp.dot(self.Wm, Z_sig)

        diff_z = Z_sig - z_pred
        weight_sqrt = jnp.sqrt(self.Wc[1])
        weighted_diff_z = diff_z[1:].T * weight_sqrt

        M = jnp.hstack([weighted_diff_z, self.R_upper.T])
        _, S_yy = lax.linalg.qr(M.T, full_matrices=False)

        def _downdate_Syy(s):
            return self._cholesky_downdate(s, diff_z[0] * jnp.sqrt(-self.Wc[0]))

        def _update_Syy(s):
            return lax.linalg.cholesky_update(s, diff_z[0] * jnp.sqrt(self.Wc[0]))

        S_yy = lax.cond(self.Wc[0] < 0, _downdate_Syy, _update_Syy, S_yy)

        diff_x = points - self.x
        weighted_diff_x = diff_x.T * self.Wc
        P_xy = weighted_diff_x @ diff_z

        Y = lax.linalg.triangular_solve(S_yy, P_xy.T, left_side=True, lower=False, transpose_a=True)
        K_T = lax.linalg.triangular_solve(S_yy, Y, left_side=True, lower=False, transpose_a=False)
        K = K_T.T

        x_new = self.x + K @ (z_point - z_pred)

        U = K @ S_yy.T

        def scan_downdate(s_carry, u_col):
            return self._cholesky_downdate(s_carry, u_col), None

        S_new, _ = lax.scan(scan_downdate, self.S, U.T)

        # 7. Return new filter instance
        return self.tree_unflatten(
            {'n_dim': self.n_dim, 'c': self.c},
            (x_new, S_new, self.Q_upper, self.R_upper, self.Wm, self.Wc)
        )
