import numpy as np
import scipy.linalg as linalg
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

from ultralytics import YOLO

from helicopter.vision import D435i
from helicopter.vision.point_detection import HelicopterYOLO, GPUImagePreprocessor
from helicopter.vision.point_detection import YOLOPointDetector
from helicopter.vision.measurement.filter_functions import transition_fn, measurement_fn
from helicopter.vision.measurement.scanner import Scanner, CameraStateHandler, PointHandler


def build_Q_matrix(dt: float, std_devs: dict) -> np.ndarray:
    I = np.eye(3)

    Q_dtheta = (std_devs['gyro'] ** 2) * I * dt

    Q_dp = std_devs['vel'] * I * dt

    Q_dv = (std_devs['accel'] ** 2) * I * dt

    Q_dba = (std_devs['bias'] ** 2) * I * dt
    Q_dbg = (std_devs['bias'] ** 2) * I * dt

    return linalg.block_diag(Q_dtheta, Q_dp, Q_dv, Q_dba, Q_dbg)


def initialize_P_matrix(std_devs: dict) -> np.ndarray:
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

    return P0


def initialize_R_matrix(std_devs: dict) -> np.ndarray:
    diag = np.array([std_devs['dp_x'] ** 2, std_devs['dp_y'] ** 2, std_devs['dp_z'] ** 2])

    return np.diag(diag)


if __name__ == '__main__':
    points = MerweScaledSigmaPoints(n=15, alpha=0.0001, beta=2., kappa=0)
    ukf = UnscentedKalmanFilter(dim_x=15, dim_z=3, dt=1 / 200,
                                hx=measurement_fn,
                                fx=transition_fn,
                                points=points)

    q_sigmas = {
        "gyro": 0.03 * (np.pi / 180.0),
        "vel": 1e-6,
        "accel": 1e-5,
        "bias": 1e-4
    }
    ukf.Q = build_Q_matrix(dt=1 / 200, std_devs=q_sigmas)

    initial_sigmas = {
        'd_theta': 0.075,
        'dp': 1e-4,
        'dv': 1e-6,
        'dba': 1.0,
        'dbg': 0.1
    }
    ukf.P = initialize_P_matrix(initial_sigmas)

    visual_sigmas = {
        'dp_x': 0.005,
        'dp_y': 0.005,
        'dp_z': 0.005,
    }
    ukf.R = initialize_R_matrix(visual_sigmas)

    device = D435i(enable_motion=True, video_rate=30,
                   projector_power=360., autoexpose=False, exposure_time=1600,
                   ema_factor=0.2)

    point_handler = PointHandler(
        detector=YOLOPointDetector(
            model=HelicopterYOLO(preprocessor=GPUImagePreprocessor(imgsz=device.IR_RESOLUTION),
                                 model=YOLO('/home/ray/yolo_models/helicopter/measure_20260203/weights/best.engine',
                                            task='detect'),
                                 conf=0.6),
            marker_tolerance=0.01,
            marker_size=0.003,
            marker_size_tolerance=0.3,
            distance_threshold=0.5
        ),
        queue_len=75)

    scanner = Scanner(device=device,
                      point_handler=point_handler,
                      camera_state_handler=CameraStateHandler(),
                      ukf=ukf,
                      measurement_time=5.0,
                      R=ukf.R.copy())

    scanner.scan()

    np.save('../../../notebooks/measured_points.npy', scanner.point_handler.points_coords)
    string_out = np.array2string(scanner.point_handler.points_coords, precision=4, separator=', ')

    print(string_out)

    print('done')
