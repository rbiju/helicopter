from abc import ABC, abstractmethod
from typing import Sequence

import cv2
import numpy as np

import pyrealsense2 as rs
from helicopter.vision.point_detection import HelicopterYOLO


class PointDetector(ABC):
    def __init__(self, marker_tolerance: float = 0.01,
                 distance_threshold: float = 0.5, ):
        self.marker_tolerance = marker_tolerance
        self.distance_threshold = distance_threshold

    @abstractmethod
    def detect(self, ir_frame: np.ndarray) -> list[cv2.KeyPoint]:
        raise NotImplementedError

    def get_points_coords(self, depth_frame, keypoints, intrinsics) -> np.ndarray:
        if len(keypoints) == 0:
            return np.empty((0, 3))

        h, w = depth_frame.shape

        valid_depths = []
        valid_centers = []
        valid_radii = []

        for kp in keypoints:
            cx, cy = kp.pt
            radius = kp.size / 2

            ix, iy = int(cx), int(cy)

            if ix < 0 or ix >= w or iy < 0 or iy >= h:
                continue

            safe_r = int(radius * 0.8)
            if safe_r < 1:
                safe_r = 1

            x0, x1 = max(0, ix - safe_r), min(w, ix + safe_r + 1)
            y0, y1 = max(0, iy - safe_r), min(h, iy + safe_r + 1)

            roi = depth_frame[y0:y1, x0:x1]

            valid_pixels = roi[roi > 0]

            if len(valid_pixels) < roi.size * 0.7:
                continue

            depth = np.mean(valid_pixels)
            d_std = np.std(valid_pixels)

            if d_std > 0.005:
                continue

            if depth > self.distance_threshold or depth <= 0:
                continue

            valid_depths.append(depth)
            valid_centers.append((cx, cy))
            valid_radii.append(radius)

        if not valid_depths:
            return np.empty((0, 3))

        depths = np.array(valid_depths)
        centers = np.array(valid_centers)

        points = []
        for center, depth in zip(centers, depths):
            point = rs.rs2_deproject_pixel_to_point(intrinsics, pixel=[center[0], center[1]], depth=depth)
            points.append(np.array([point[2], -point[0], -point[1]]))

        return np.vstack(points)


class BlobPointDetector(PointDetector):
    def __init__(self, marker_tolerance: float = 0.01,
                 distance_threshold: float = 0.5):
        super().__init__(marker_tolerance, distance_threshold)

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
                 distance_threshold: float = 0.5):
        super().__init__(marker_tolerance, distance_threshold)
        self.model = model

    @staticmethod
    def get_refined_keypoints(ir_frame, boxes, margin=2) -> Sequence[cv2.KeyPoint]:
        if len(boxes) == 0:
            return []

        h, w = ir_frame.shape

        x1 = np.clip(boxes[:, 0] - margin, 0, w).astype(int)
        y1 = np.clip(boxes[:, 1] - margin, 0, h).astype(int)
        x2 = np.clip(boxes[:, 2] + margin, 0, w).astype(int)
        y2 = np.clip(boxes[:, 3] + margin, 0, h).astype(int)

        keypoints = []

        for i in range(len(boxes)):
            roi = ir_frame[y1[i]:y2[i], x1[i]:x2[i]]

            if roi.size == 0:
                continue

            _, roi_clean = cv2.threshold(roi, 60, 255, cv2.THRESH_TOZERO)

            M = cv2.moments(roi_clean, binaryImage=False)

            if M["m00"] <= 0:
                continue

            cx = x1[i] + (M["m10"] / M["m00"])
            cy = y1[i] + (M["m01"] / M["m00"])

            _radius = np.sqrt(M["m00"] / 255.0 / np.pi)

            keypoints.append(cv2.KeyPoint(x=float(cx), y=float(cy), size=float(_radius)))

        return keypoints

    def detect(self, ir_frame: np.ndarray) -> Sequence[cv2.KeyPoint]:
        boxes = self.model(ir_frame)
        keypoints = self.get_refined_keypoints(ir_frame, boxes, margin=1)
        return keypoints
