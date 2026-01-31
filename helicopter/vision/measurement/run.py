import numpy as np
import scipy.linalg as linalg
from filterpy.kalman import UnscentedKalmanFilter, MerweScaledSigmaPoints

from ultralytics import YOLO

from helicopter.vision import D435i
from helicopter.vision.point_detection import HelicopterYOLO, GPUImagePreprocessor
from helicopter.vision.point_detection.measurement import YOLOPointDetector
from helicopter.vision.measurement.filter_functions import transition_fn, measurement_fn
from helicopter.vision.measurement.scanner import Scanner, CameraStateHandler, PointHandler, MadgwickFilter


def build_Q_matrix(dt: float) -> np.ndarray:
    sigma_arw = 0.028 * (np.pi / 180.0)

    g_m_s2 = 9.81
    sigma_vrw = 150e-6 * g_m_s2

    sigma_bgrw = 1.0e-4
    sigma_barw = 1.0e-4
    I = np.eye(3)

    Q_dtheta = (sigma_arw ** 2) * I * dt
    Q_dp = 1.0e-5 * I * dt

    Q_dv = (sigma_vrw ** 2) * I * dt

    Q_dba = (sigma_barw ** 2) * I * dt

    Q_dbg = (sigma_bgrw ** 2) * I * dt

    Q = linalg.block_diag(Q_dtheta, Q_dp, Q_dv, Q_dba, Q_dbg)

    return Q


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
    std_q = std_devs['d_theta_vis'] / 2
    var_q = std_q ** 2
    var_p = std_devs['dp_vis'] ** 2

    diag = np.array([var_q, var_q, var_q, var_p, var_p, var_p])

    return np.diag(diag)


if __name__ == '__main__':
    points = MerweScaledSigmaPoints(n=15, alpha=0.1, beta=2., kappa=0.)
    ukf = UnscentedKalmanFilter(dim_x=15, dim_z=6, dt=1 / 200,
                                hx=measurement_fn,
                                fx=transition_fn,
                                points=points)

    ukf.Q = build_Q_matrix(dt=1 / 200)

    initial_sigmas = {
        'd_theta': 0.075,
        'dp': 0.005,
        'dv': 0.001,
        'dba': 0.5,
        'dbg': 0.5
    }
    ukf.P = initialize_P_matrix(initial_sigmas)

    visual_sigmas = {
        'd_theta_vis': 0.04,
        'dp_vis': 0.01
    }
    ukf.R = initialize_R_matrix(visual_sigmas)

    device = D435i(enable_motion=True, projector_power=360., autoexpose=False, exposure_time=1600,
                   ema_factor=0.15)

    point_handler = PointHandler(
        detector=YOLOPointDetector(
            model=HelicopterYOLO(preprocessor=GPUImagePreprocessor(imgsz=device.IR_RESOLUTION),
                                 model=YOLO('/home/ray/yolo_models/helicopter/measure/weights/best.engine',
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
                      madgwick=MadgwickFilter(beta=0.033),
                      measurement_time=10.0)

    scanner.scan()

    np.save('../../../notebooks/measured_points.npy', scanner.point_handler.points_coords)
    string_out = np.array2string(scanner.point_handler.points_coords, precision=4, separator=', ')

    print(string_out)

    print('done')
