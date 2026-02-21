import jax
import jax.numpy as jnp
import numpy as np
import scipy.linalg as linalg
from ultralytics import YOLO

from helicopter.vision import D435i
from helicopter.vision import ErrorStateSquareRootUnscentedKalmanFilter as UKF
from helicopter.vision.point_detection import HelicopterYOLO, GPUImagePreprocessor
from helicopter.vision.point_detection import YOLOPointDetector
from helicopter.vision.measurement.scanner import Scanner, CameraStateHandler, PointHandler


def initialize_Q_matrix(dt: float, std_devs: dict) -> np.ndarray:
    I = np.eye(3)

    Q_dtheta = (std_devs['gyro'] ** 2) * I * dt

    Q_dp = (std_devs['pos'] ** 2) * I * dt

    Q_dv = (std_devs['vel'] ** 2) * I * dt

    Q_dba = (std_devs['bias_acc'] ** 2) * I * dt
    Q_dbg = (std_devs['bias_gyro'] ** 2) * I * dt

    return linalg.block_diag(Q_dtheta, Q_dp, Q_dv, Q_dba, Q_dbg)


def initialize_S_matrix(std_devs: dict) -> jax.Array:
    var_dtheta = std_devs['d_theta'] ** 2
    P_dtheta = np.eye(3) * var_dtheta

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


if __name__ == '__main__':
    N = 15
    q_sigmas = {
        "gyro": 0.05 * (np.pi / 180.0),
        "pos": 1e-5,
        "vel": 1e-3,
        "bias_acc": 1e-7,
        "bias_gyro": 1e-7
    }

    Q = initialize_Q_matrix(dt=1 / 200, std_devs=q_sigmas)

    initial_sigmas = {
        'd_theta': 0.05 * (np.pi / 180.0),
        'dp': 1e-5,
        'dv': 1e-4,
        'dba': 1e-5,
        'dbg': 1e-5,
    }
    S = initialize_S_matrix(initial_sigmas)

    visual_sigmas = {
        'dp_x': 5e-3,
        'dp_y': 5e-3,
        'dp_z': 5e-3,
    }
    R = initialize_R_matrix(visual_sigmas)
    x = jnp.zeros(N)
    ukf = UKF(x=x, S=S, Q=Q, R=R, alpha=0.1, beta=2.0, kappa=-12)

    device = D435i(enable_motion=True, video_rate=60,
                   projector_power=360., autoexpose=False, exposure_time=2200,
                   ema_accel=0.75,
                   ema_gyro=0.75,)

    point_handler = PointHandler(
        detector=YOLOPointDetector(
            model=HelicopterYOLO(preprocessor=GPUImagePreprocessor(imgsz=device.IR_RESOLUTION),
                                 model=YOLO('/home/ray/yolo_models/helicopter/measure_20260203/weights/best.engine',
                                            task='detect'),
                                 conf=0.65),
            marker_tolerance=0.01,
            marker_size=0.003,
            marker_size_tolerance=0.75,
            distance_threshold=0.4
        ),
        queue_len=75)

    scanner = Scanner(device=device,
                      point_handler=point_handler,
                      camera_state_handler=CameraStateHandler(),
                      ukf=ukf,
                      measurement_time=30.0)

    scanner.scan()
