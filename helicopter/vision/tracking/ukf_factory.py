import jax
import jax.numpy as jnp

from helicopter.vision import ErrorStateSquareRootUnscentedKalmanFilter


def initialize_Q_matrix(dt: float, std_devs: dict) -> jax.Array:
    I = jnp.eye(3)

    Q_dtheta = ((std_devs['orientation'] * jnp.pi / 180) ** 2) * I * dt
    Q_dp = (std_devs['position'] ** 2) * I * dt
    Q_do = ((std_devs['angular_velocity'] * jnp.pi / 180) ** 2) * I * dt
    Q_dv = (std_devs['velocity'] ** 2) * I * dt
    Q_commands = (std_devs['commands'] ** 2) * I * dt
    Q_battery = std_devs['battery'] ** 2 * dt
    Q_trim = std_devs['trim'] ** 2 * dt

    return jax.scipy.linalg.block_diag(Q_dtheta, Q_dp, Q_do, Q_dv, Q_commands, Q_battery, Q_trim)


def initialize_S_matrix(std_devs: dict) -> jax.Array:
    I = jnp.eye(3)

    P_dtheta = ((std_devs['orientation'] * jnp.pi / 180) ** 2) * I
    P_dp = (std_devs['position'] ** 2) * I
    P_do = ((std_devs['angular_velocity'] * jnp.pi / 180) ** 2) * I
    P_dv = (std_devs['velocity'] ** 2) * I
    P_commands = (std_devs['commands'] ** 2) * I
    P_battery = std_devs['battery'] ** 2
    P_trim = std_devs['trim'] ** 2

    P0 = jax.scipy.linalg.block_diag(P_dtheta, P_dp, P_do, P_dv, P_commands, P_battery, P_trim)
    _S = jax.lax.linalg.cholesky(jnp.array(P0)).T

    return _S


def initialize_R_matrix(std_devs: dict) -> jax.Array:
    diag = jnp.array([std_devs['dp_x'] ** 2, std_devs['dp_y'] ** 2, std_devs['dp_z'] ** 2])

    return jnp.diag(diag)


class TrackerUKFFactory:
    def __init__(self,
                 dt: float,
                 q_std_devs: dict,
                 s_std_devs: dict,
                 r_std_devs: dict,
                 N: int = 17,
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
        cpu_device = jax.devices('cpu')[0]
        with jax.default_device(cpu_device):
            ukf = ErrorStateSquareRootUnscentedKalmanFilter(x=self.x, S=self.S, Q=self.Q, R=self.R,
                                                         alpha=self.alpha, beta=self.beta, kappa=self.kappa)
        return ukf
