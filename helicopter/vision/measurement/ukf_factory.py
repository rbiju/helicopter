import jax
import jax.numpy as jnp

from helicopter.vision import ErrorStateSquareRootUnscentedKalmanFilter


def initialize_Q_matrix(dt: float, std_devs: dict) -> jax.Array:
    I = jnp.eye(3)

    Q_dtheta = ((std_devs['gyro'] * jnp.pi / 180) ** 2) * I * dt

    Q_dp = (std_devs['pos'] ** 2) * I * dt

    Q_dv = (std_devs['vel'] ** 2) * I * dt

    Q_dba = (std_devs['bias_acc'] ** 2) * I * dt
    Q_dbg = (std_devs['bias_gyro'] ** 2) * I * dt

    return jax.scipy.linalg.block_diag(Q_dtheta, Q_dp, Q_dv, Q_dba, Q_dbg)


def initialize_S_matrix(std_devs: dict) -> jax.Array:
    var_dtheta = (std_devs['d_theta'] * jnp.pi / 180.) ** 2
    P_dtheta = jnp.eye(3) * var_dtheta

    var_dp = std_devs['dp'] ** 2
    P_dp = jnp.eye(3) * var_dp

    var_dv = std_devs['dv'] ** 2
    P_dv = jnp.eye(3) * var_dv

    var_dba = std_devs['dba'] ** 2
    P_dba = jnp.eye(3) * var_dba

    var_dbg = std_devs['dbg'] ** 2
    P_dbg = jnp.eye(3) * var_dbg

    P0 = jax.scipy.linalg.block_diag(P_dtheta, P_dp, P_dv, P_dba, P_dbg)
    _S = jax.lax.linalg.cholesky(jnp.array(P0)).T

    return _S


def initialize_R_matrix(std_devs: dict) -> jax.Array:
    diag = jnp.array([std_devs['dp_x'] ** 2, std_devs['dp_y'] ** 2, std_devs['dp_z'] ** 2])

    return jnp.diag(diag)


class MeasurementUKFFactory:
    def __init__(self, dt: float,
                 q_std_devs: dict,
                 s_std_devs: dict,
                 r_std_devs: dict,
                 N: int = 15,
                 alpha: float = 0.1,
                 beta: float = 2.0,
                 kappa: float = -12):

        self.Q = initialize_Q_matrix(dt=dt, std_devs=q_std_devs)
        self.S = initialize_S_matrix(std_devs=s_std_devs)
        self.R = initialize_R_matrix(std_devs=r_std_devs)
        self.x = jnp.zeros((N,))

        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

    def filter(self) -> ErrorStateSquareRootUnscentedKalmanFilter:
        return ErrorStateSquareRootUnscentedKalmanFilter(x=self.x, S=self.S, Q=self.Q, R=self.R,
                                                         alpha=self.alpha, beta=self.beta, kappa=self.kappa)