from abc import ABC, abstractmethod
from typing import Sequence

import cv2
import numpy as np

import pyrealsense2 as rs

from helicopter.vision.point_detection import HelicopterYOLO


class PointDetector(ABC):
    def __init__(self, marker_tolerance: float = 0.01,
                 marker_size: float = 0.003,
                 marker_size_tolerance: float = 0.3,
                 distance_threshold: float = 0.5,):
        self.marker_tolerance = marker_tolerance
        self.marker_size = marker_size
        self.marker_size_tolerance = marker_size_tolerance
        self.distance_threshold = distance_threshold

    @abstractmethod
    def detect(self, ir_frame: np.ndarray) -> list[cv2.KeyPoint]:
        raise NotImplementedError

    @staticmethod
    def draw_subpixel_circle(center, radius, shift):
        factor = 1 << shift

        cx, cy = center

        cx_rounded = round(cx * factor)
        cy_rounded = round(cy * factor)
        radius_rounded = round(radius * factor)

        return (cx_rounded, cy_rounded), radius_rounded

    def get_point_coord(self, depth_frame, circle, intrinsics: rs.intrinsics) -> np.ndarray | None:
        h, w = depth_frame.shape
        center, radius, _ = circle

        safe_radius = radius * 0.8

        ix, iy = int(center[0]), int(center[1])
        margin = int(safe_radius + 2)

        x0, x1 = max(0, ix - margin), min(w, ix + margin + 1)
        y0, y1 = max(0, iy - margin), min(h, iy + margin + 1)

        depth_roi = depth_frame[y0:y1, x0:x1]

        y_grid, x_grid = np.ogrid[y0:y1, x0:x1]
        dist_sq = (x_grid - center[0]) ** 2 + (y_grid - center[1]) ** 2
        mask = dist_sq <= safe_radius ** 2

        valid_pixels = depth_roi[mask & (depth_roi > 0)]

        if len(valid_pixels) < 5:
            return None

        depth_median = np.median(valid_pixels)

        noise_tolerance = 0.02
        clean_pixels = valid_pixels[np.abs(valid_pixels - depth_median) < noise_tolerance]

        if len(clean_pixels) == 0:
            return None

        depth = np.mean(clean_pixels)

        if depth > self.distance_threshold:
            return None

        physical_diameter = (radius * 2 * depth) / intrinsics.fx

        if abs(physical_diameter - self.marker_size) > (self.marker_size * self.marker_size_tolerance):
            return None

        point = rs.rs2_deproject_pixel_to_point(intrinsics, pixel=[center[0], center[1]], depth=depth)

        return np.array([point[2], -point[0], -point[1]])

    def get_point_coords(self, keypoints: list[cv2.KeyPoint], depth_frame: np.ndarray, intrinsics: rs.intrinsics,
                         shift: int = 3) -> np.ndarray:
        points = []
        for kp in keypoints:
            marker_point = self.get_point_coord(depth_frame,
                                                (kp.pt, kp.size / 2, shift),
                                                intrinsics)

            if marker_point is None:
                continue

            points.append(marker_point)

        return np.array(points)


class BlobPointDetector(PointDetector):
    def __init__(self, marker_tolerance: float = 0.01,
                 marker_size: float = 0.003,
                 marker_size_tolerance: float = 0.3,
                 distance_threshold: float = 0.5):
        super().__init__(marker_tolerance, marker_size, marker_size_tolerance, distance_threshold)

        params = cv2.SimpleBlobDetector.Params()

        params.filterByArea = True
        params.minArea = 5
        params.maxArea = 30

        params.filterByColor = True
        params.blobColor = 255

        params.filterByInertia = True
        params.minInertiaRatio = 0.6

        params.filterByCircularity = True
        params.minCircularity = 0.75

        params.filterByConvexity = True
        params.minConvexity = 0.9

        params.thresholdStep = 15
        params.minThreshold = 40
        params.maxThreshold = 200

        self.detector = cv2.SimpleBlobDetector.create(params)
        self.tophat_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
        self.clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(25, 25))

    def detect(self, ir_frame: np.ndarray) -> Sequence[cv2.KeyPoint]:
        tophat = cv2.morphologyEx(ir_frame, cv2.MORPH_TOPHAT, self.tophat_kernel)
        clahe = self.clahe.apply(tophat)

        keypoints = self.detector.detect(clahe)

        return keypoints


class YOLOPointDetector(PointDetector):
    def __init__(self,
                 model: HelicopterYOLO,
                 marker_tolerance: float = 0.01,
                 marker_size: float = 0.003,
                 marker_size_tolerance: float = 0.3,
                 distance_threshold: float = 0.5):
        super().__init__(marker_tolerance, marker_size, marker_size_tolerance, distance_threshold)
        self.model = model

    def detect(self, ir_frame: np.ndarray) -> Sequence[cv2.KeyPoint]:
        results = self.model(ir_frame)

        boxes = results[0].boxes.data.clone()
        boxes[:, [1, 3]] -= self.model.preprocessor.top_pad

        boxes = boxes[:, :4].cpu().numpy().astype(int)

        margin = 1
        keypoints = []
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            h, w = ir_frame.shape

            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)

            roi = ir_frame[y1:y2, x1:x2]

            if roi.size == 0:
                continue
            _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours:
                continue

            largest_contour = max(contours, key=cv2.contourArea)

            (cx_roi, cy_roi), radius = cv2.minEnclosingCircle(largest_contour)

            final_x = x1 + cx_roi
            final_y = y1 + cy_roi

            keypoints.append(cv2.KeyPoint(x=final_x, y=final_y, size=radius * 2))

        return keypoints
