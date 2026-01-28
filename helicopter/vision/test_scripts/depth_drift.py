import time
import numpy as np
import cv2

import pyrealsense2 as rs

from helicopter.vision import D435i
from helicopter.vision.utils import PointQueue


def snapshot(_camera: D435i):
    depth_image = None
    ir_image = None

    snapped = False
    while not snapped:
        frames = _camera.pipeline.poll_for_frames()
        depth_frame = frames.get_depth_frame()
        ir_frame = frames.get_infrared_frame()
        if depth_frame and ir_frame:
            depth_image, _, ir_image, _, _ = _camera.process_frames(frames)

            snapped = True
            print("Snapshot retrieved.")

    return depth_image, ir_image


def draw_subpixel_circle(center, radius, shift=4):
    factor = 1 << shift

    cx, cy = center

    cx_rounded = round(cx * factor)
    cy_rounded = round(cy * factor)
    radius_rounded = round(radius * factor)

    return (cx_rounded, cy_rounded), radius_rounded


def detect_points(ir_frame, depth_frame, intrinsics):
    valid_mask = 0 < depth_frame

    params = cv2.SimpleBlobDetector.Params()

    params.filterByArea = True
    params.minArea = 1
    params.maxArea = 30

    params.filterByColor = True
    params.blobColor = 255

    params.filterByInertia = True
    params.minInertiaRatio = 0.7

    params.filterByCircularity = True
    params.minCircularity = 0.75

    params.filterByConvexity = True
    params.minConvexity = 0.9

    params.thresholdStep = 20
    params.minThreshold = 20
    params.maxThreshold = 180

    detector = cv2.SimpleBlobDetector.create(params)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 5))
    clahe_operator = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(25, 25))

    start = time.perf_counter()
    ir_cleaned = cv2.medianBlur(ir_frame, 3)
    tophat = cv2.morphologyEx(ir_cleaned, cv2.MORPH_TOPHAT, kernel)
    clahe = clahe_operator.apply(tophat)

    keypoints = detector.detect(clahe)

    filtered_keypoints = []
    points = []
    shift = 4
    h, w = depth_frame.shape
    for kp in keypoints:
        r_pixel = kp.size / 2
        ix, iy = int(kp.pt[0]), int(kp.pt[1])
        margin = int(r_pixel + 2)

        x0, x1 = max(0, ix - margin), min(w, ix + margin + 1)
        y0, y1 = max(0, iy - margin), min(h, iy + margin + 1)

        depth_roi = depth_frame[y0:y1, x0:x1]
        valid_roi = valid_mask[y0:y1, x0:x1]

        roi_h, roi_w = depth_roi.shape
        local_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)

        local_center = (kp.pt[0] - x0, kp.pt[1] - y0)

        c_sub, r_sub = draw_subpixel_circle(local_center, r_pixel, shift)
        cv2.circle(local_mask, c_sub, r_sub, 1, -1, shift=shift)

        ksize = int(r_pixel) | 1
        gaussian_roi = cv2.GaussianBlur(local_mask.astype(float), (ksize, ksize),
                                        r_pixel * 0.9) * valid_roi

        g_sum = gaussian_roi.sum()
        if g_sum <= 0:
            continue

        depth = np.sum(depth_roi * (gaussian_roi / g_sum))
        if np.isnan(depth):
            continue
        if depth > 0.5:
            continue

        physical_diameter = (r_pixel * 2 * depth) / intrinsics.fx
        if physical_diameter > 0.003 * (1 + 0.5) or physical_diameter < 0.003 * 0.5:
            continue

        filtered_keypoints.append(kp)

        point = rs.rs2_deproject_pixel_to_point(intrinsics, pixel=[kp.pt[0], kp.pt[1]], depth=depth)
        points.append(np.array([point[2], -point[0], -point[1]]))

    end = time.perf_counter()

    print(f"Detect time: {end - start}")

    final_detection = cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR)
    for kp in filtered_keypoints:
        cv2.circle(final_detection, (int(kp.pt[0]), int(kp.pt[1])), int(kp.size), (0, 255, 0), 1)

    if len(points) > 3:
        return np.vstack(points), final_detection
    else:
        return None


def register_points(_point_collection: list[np.ndarray]):
    _registered_points = {}
    next_id = 0

    def is_registered(_point):
        if len(_registered_points) < 1:
            return None

        point_list = [pq.mean for pq in _registered_points.values()]
        registered_points_coords = np.array(point_list)

        norm = np.linalg.norm(registered_points_coords - _point, axis=1)
        comp = norm < 0.01
        if np.any(comp):
            return int(np.argmin(norm))
        else:
            return None

    for points in _point_collection:
        for point in points:
            registered_idx = is_registered(point)
            if registered_idx is None:
                _registered_points[next_id] = PointQueue(maxlen=15, init_value=point)
                next_id += 1
            else:
                _registered_points[registered_idx].enqueue(point)

    return _registered_points


if __name__ == '__main__':
    camera = D435i(projector_power=360.,
                   autoexpose=False,
                   exposure_time=2000)

    final_detections = []
    point_collection = []
    for i in range(100):
        _depth_image, _ir_image = snapshot(camera)
        detect_out = detect_points(_ir_image, _depth_image, camera.intrinsics)
        if detect_out is None:
            continue
        else:
            _points, _detection = detect_out
            print(i)
            print(_points)
            point_collection.append(_points)
            final_detections.append(_detection)

    camera.stop()

    start_register = time.perf_counter()
    registered_points = register_points(point_collection)
    end_register = time.perf_counter()

    print(f"Register time: {end_register-start_register}")
    out = np.vstack([pq.mean for pq in registered_points.values()])
    string_out = np.array2string(out, precision=4, separator=', ')
    print(string_out)

    print('done')
