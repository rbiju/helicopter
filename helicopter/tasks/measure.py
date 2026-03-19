import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as linalg

from helicopter.configuration import HydraConfigurable
from helicopter.vision import D435i
from helicopter.vision import ErrorStateSquareRootUnscentedKalmanFilter as UKF
from helicopter.vision.measurement.scanner import Scanner, CameraStateHandler, MeasurementPointHandler

from .base import Task


def initialize_Q_matrix(dt: float, std_devs: dict) -> np.ndarray:
    I = np.eye(3)

    Q_dtheta = ((std_devs['gyro'] * np.pi / 180) ** 2) * I * dt

    Q_dp = (std_devs['pos'] ** 2) * I * dt

    Q_dv = (std_devs['vel'] ** 2) * I * dt

    Q_dba = (std_devs['bias_acc'] ** 2) * I * dt
    Q_dbg = (std_devs['bias_gyro'] ** 2) * I * dt

    return linalg.block_diag(Q_dtheta, Q_dp, Q_dv, Q_dba, Q_dbg)


def initialize_S_matrix(std_devs: dict) -> jax.Array:
    P_dtheta = (std_devs['d_theta'] * np.pi / 180.) ** 2

    var_dp = std_devs['dp'] ** 2
    P_dp = np.eye(3) * var_dp

    var_dv = std_devs['dv'] ** 2
    P_dv = np.eye(3) * var_dv

    var_dba = std_devs['dba'] ** 2
    P_dba = np.eye(3) * var_dba

    var_dbg = std_devs['dbg'] ** 2
    P_dbg = np.eye(3) * var_dbg

    P0 = linalg.block_diag(P_dtheta, P_dp, P_dv, P_dba, P_dbg)
    _S = jax.lax.linalg.cholesky(jnp.array(P0)).T

    return _S


def initialize_R_matrix(std_devs: dict) -> np.ndarray:
    diag = np.array([std_devs['dp_x'] ** 2, std_devs['dp_y'] ** 2, std_devs['dp_z'] ** 2])

    return np.diag(diag)


class UKFFactory:
    def __init__(self, dt: float,
                 q_std_devs: dict,
                 s_std_devs: dict,
                 r_std_devs: dict,
                 N: int,
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

    def filter(self) -> UKF:
        return UKF(x=self.x, S=self.S, Q=self.Q, R=self.R,
                   alpha=self.alpha, beta=self.beta, kappa=self.kappa)


@HydraConfigurable
class Measure(Task):
    def __init__(self,
                 ukf_factory: UKFFactory,
                 device: D435i,
                 point_handler: MeasurementPointHandler,
                 measurement_time: float = 30.0):
        super().__init__()

        self.scanner = Scanner(device=device,
                               point_handler=point_handler,
                               camera_state_handler=CameraStateHandler(),
                               ukf=ukf_factory.filter(),
                               measurement_time=measurement_time)


    def run(self, configuration_path: str):
        self.scanner.scan()
