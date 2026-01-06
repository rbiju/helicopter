import cv2
import numpy as np
import pyrealsense2 as rs


class PointGetter:
    def get_ellipse_coords(self, depth_frame, ellipse, intrinsics: rs.intrinsics, depth_scale: float,
                           sigma_factor=0.3) -> np.ndarray:
        (cx, cy), (MA, ma), angle = ellipse

        depth_frame = depth_frame * depth_scale
        valid_mask = depth_frame > 0

        mask = np.zeros(depth_frame.shape[:2], dtype=np.uint8)
        cv2.ellipse(mask, ellipse, 255, -1)

        ksize = int(max(MA, ma)) | 1
        sigma = ksize * sigma_factor
        gaussian_mask = cv2.GaussianBlur(mask.astype(float), (ksize, ksize), sigma) * valid_mask

        gaussian_mask = gaussian_mask / gaussian_mask.sum()
        depth = np.sum(depth_frame * gaussian_mask)

        return np.array(rs.rs2_deproject_pixel_to_point(intrinsics, pixel=[cx, cy], depth=depth))

    @staticmethod
    def retrieve_ellipses(ir_frame: np.ndarray, gof_threshold: float, edge_min: float = 500., edge_max: float = 600) -> list[cv2.typing.RotatedRect]:
        edges = cv2.Canny(ir_frame, edge_min, edge_max)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ellipses = []
        for contour in contours:
            if len(contour) >= 5:
                ellipse = cv2.fitEllipse(contour)
                (_, _), (MA, ma), angle = ellipse
                if cv2.contourArea(contour) > 0:
                    if np.abs(1. - ((np.pi * (MA / 2.) * (ma / 2.)) / cv2.contourArea(contour))) < gof_threshold:
                        ellipses.append(ellipse)

        return ellipses

    def get_points(self, ir_frame: np.ndarray,
                   depth_frame: np.ndarray, intrinsics: rs.intrinsics, depth_scale: float,
                   edge_min: float = 500, edge_max: float = 600, gof_threshold: float = 0.30) -> np.ndarray:
        ellipses = self.retrieve_ellipses(ir_frame, gof_threshold, edge_min, edge_max)

        points = np.empty((len(ellipses), 3), dtype=np.float32)
        for i, ellipse in enumerate(ellipses):
            points[i] = self.get_ellipse_coords(depth_frame, ellipse, intrinsics, depth_scale)

        return points
