import numpy as np
import quaternion
import cv2

from filterpy.kalman import UnscentedKalmanFilter

import pyrealsense2 as rs

from .utils import PointQueue


class PointHandler:
    def __init__(self, ir_threshold: int = 220,
                 marker_radius: float = 0.003,
                 frame_center: tuple[int, int] = (640, 360),
                 fov: tuple[float, float] = (87., 58.),
                 maxlen: int = 60) -> None:
        self.ir_threshold = ir_threshold
        self.marker_radius = marker_radius
        self.frame_center = frame_center
        self.fov = fov

        self.maxlen = maxlen

        self.degrees_per_pixel = (self.fov[0] / (self.frame_center[0] * 2),
                                  self.fov[1] / (self.frame_center[1] * 2))

        self.points: dict[int, PointQueue] = {}
        self._points_coords = None

    def is_point_registered(self, point: np.ndarray) -> bool | int:
        if len(self.points) < 1:
            return False

        norm = np.linalg.norm(self._points_coords - point, axis=1)
        comp = norm < self.marker_radius
        if np.any(comp):
            return int(np.argmax(comp))
        else:
            return False

    def get_ellipse_coords(self, depth_frame, ellipse, sigma_factor=0.3) -> np.ndarray:
        (cx, cy), (MA, ma), angle = ellipse
        cx, cy = int(cx), int(cy)

        mask = np.zeros(depth_frame.shape[:2], dtype=np.uint8)
        cv2.ellipse(mask, ellipse, 255, -1)

        ksize = int(max(MA, ma)) | 1
        sigma = ksize * sigma_factor
        gaussian_mask = cv2.GaussianBlur(mask.astype(float), (ksize, ksize), sigma)

        gaussian_mask = gaussian_mask / gaussian_mask.sum()
        weighted_avg = np.sum(depth_frame * gaussian_mask)

        return np.ndarray(
            [np.sin(self.degrees_per_pixel[0] * (cx - self.frame_center[0]) * (np.pi / 180)) * weighted_avg,
             np.sin(self.degrees_per_pixel[1] * (cy - self.frame_center[1]) * (np.pi / 180)) * weighted_avg,
             weighted_avg])

    def get_points(self, ir_frame: np.ndarray, depth_frame: np.ndarray):
        blurred = cv2.GaussianBlur(ir_frame, (3, 3), 0)
        _, thresh = cv2.threshold(blurred, self.ir_threshold, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                ellipse_point = self.get_ellipse_coords(depth_frame, ellipse)
                registered = self.is_point_registered(ellipse_point)
                if not registered:
                    self.points[len(self.points)] = PointQueue(maxlen=self.maxlen, init_value=ellipse_point)
                else:
                    self.points[registered].enqueue(ellipse_point)

        self._points_coords = np.vstack(tuple([point_queue.mean for point_queue in self.points.values()]))


class Camera:
    def __init__(self, gyro_dt: float, accel_dt: float,):
        self.position = np.array([0.0, 0.0, 0.0])
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.quaternion = quaternion.quaternion(1.0, 0.0, 0.0, 0.0)

        self.g = np.array([0., 0., 9.80665])

        self.accelerometer_bias = np.array([0.0, 0.0, 0.0])
        self.gyro_bias = np.array([0.0, 0.0, 0.0])

        self.gyro_dt = gyro_dt
        self.accel_dt = accel_dt

    def get_imu_pose(self, accelerometer, gyro):
        gyro = gyro - self.gyro_bias
        accelerometer = accelerometer - self.accelerometer_bias

        # gyro integration
        gyro_norm = np.linalg.norm(gyro)
        if gyro_norm < 1e-6:
            dq = quaternion.quaternion(1.0, 0.0, 0.0, 0.0)
        else:
            theta_half = (gyro_norm * self.gyro_dt) / 2.
            sin_theta_half = np.sin(theta_half)
            cos_theta_half = np.cos(theta_half)
            vec = gyro / gyro_norm * sin_theta_half
            dq = quaternion.quaternion([cos_theta_half, *vec])

        self.quaternion = self.quaternion * dq
        self.quaternion = self.quaternion.normalized()

        # accelerometer orientation correction
        acc_quat = quaternion.quaternion(0, *accelerometer)
        acc_rotated = (self.quaternion * acc_quat * self.quaternion.conjugate()).imag
        acc_adjusted = acc_rotated - self.g

        self.velocity = self.velocity + (acc_adjusted * self.accel_dt)
        self.position = self.position + (self.velocity * self.accel_dt)

    def get_visual_pose(self, measured_points: np.ndarray, reference_points: np.ndarray):
        m_c = measured_points.mean(axis=0)
        r_c = reference_points.mean(axis=0)

        measured_points_centered = measured_points - m_c
        reference_points_centered = reference_points - r_c

        covar = measured_points_centered.transpose() @ reference_points_centered
        U, s, Vh = np.linalg.svd(covar)

        rotation_matrix = U @ Vh
        translation = r_c - m_c

        return translation, quaternion.from_rotation_matrix(rotation_matrix)


class MeasurementTool:
    def __init__(self, point_handler: PointHandler, camera: Camera, ukf: UnscentedKalmanFilter):
        self.point_handler = point_handler
        self.camera = camera
        self.ukf = ukf

    def initialize(self):
        pass



