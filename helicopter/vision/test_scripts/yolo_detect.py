import time

import cv2
import numpy as np
import pyrealsense2 as rs

from ultralytics import YOLO

from helicopter.vision import D435i
from helicopter.vision.point_detection import HelicopterYOLO


def draw_subpixel_circle(center, _radius, _shift=4):
    factor = 1 << _shift

    _cx, _cy = center

    cx_rounded = round(_cx * factor)
    cy_rounded = round(_cy * factor)
    radius_rounded = round(_radius * factor)

    return (cx_rounded, cy_rounded), radius_rounded


if __name__ == '__main__':
    camera = D435i(projector_power=360.,
                   autoexpose=False,
                   exposure_time=2000)
    ir_image = None
    depth_image = None
    frame_count = 0
    while frame_count < 1:
        frames = camera.pipeline.wait_for_frames()
        depth_image, ts_depth, ir_image, ts_ir, laser_state = camera.process_frames(frames)
        frame_count += 1

    camera.stop()

    model = HelicopterYOLO(model=YOLO('/home/ray/yolo_models/helicopter/measure/weights/best.engine',
                                      task='detect'),
                           conf=0.6)

    if ir_image is not None:
        start_detect = time.perf_counter()
        results = model(ir_image)
        end_inference = time.perf_counter()

        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        boxes[:, [1, 3]] -= model.preprocessor.top_pad

        circles = []
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            h, w = ir_image.shape

            # 1. Safe Crop (Handle edges)
            x1 = max(0, x1 - 1)
            y1 = max(0, y1 - 1)
            x2 = min(w, x2 + 1)
            y2 = min(h, y2 + 1)

            roi = ir_image[y1:y2, x1:x2]

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

            circles.append(cv2.KeyPoint(x=final_x, y=final_y, size=radius * 2))

        end_detect = time.perf_counter()
        detect_time = end_detect - start_detect

        valid_mask = depth_image > 0
        filtered_keypoints = []
        points = []
        shift = 4
        h, w = depth_image.shape
        for kp in circles:
            r_pixel = kp.size / 2
            ix, iy = int(kp.pt[0]), int(kp.pt[1])
            margin = int(r_pixel + 0.5)

            x0, x1 = max(0, ix - margin), min(w, ix + margin + 1)
            y0, y1 = max(0, iy - margin), min(h, iy + margin + 1)

            depth_roi = depth_image[y0:y1, x0:x1]
            valid_roi = valid_mask[y0:y1, x0:x1]

            if np.sum(valid_roi) < (0.5 * (np.pi * r_pixel ** 2)):
                continue

            roi_h, roi_w = depth_roi.shape
            local_mask = np.zeros((roi_h, roi_w), dtype=np.uint8)

            local_center = (kp.pt[0] - x0, kp.pt[1] - y0)

            c_sub, r_sub = draw_subpixel_circle(local_center, r_pixel, shift)
            cv2.circle(local_mask, c_sub, r_sub, 1, -1, shift=shift)

            ksize = max(int(r_pixel) | 1, 3)
            gaussian_roi = cv2.GaussianBlur(local_mask.astype(float), (ksize, ksize),
                                            r_pixel * 0.9) * valid_roi

            g_sum = gaussian_roi.sum()
            if g_sum <= 0:
                continue

            depth_std = depth_roi.std()
            if depth_std > 0.003:
                continue

            depth = np.sum(depth_roi * gaussian_roi / g_sum)
            if np.isnan(depth):
                continue
            if depth > 0.5:
                continue

            physical_diameter = (r_pixel * 2 * depth) / camera.intrinsics.fx
            if physical_diameter > 0.003 * (1 + 0.5) or physical_diameter < 0.003 * 0.5:
                continue

            filtered_keypoints.append(kp)

            point = rs.rs2_deproject_pixel_to_point(camera.intrinsics, pixel=[kp.pt[0], kp.pt[1]], depth=depth)
            points.append(np.array([point[2], -point[0], -point[1]]))

        end_point_get = time.perf_counter()
        point_get = end_point_get - start_detect

        display_image = ir_image.copy()
        display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)
        for kp in filtered_keypoints:
            (cx, cy) = kp.pt
            radius = kp.size / 2
            # c_sub, r_sub = draw_subpixel_circle((cx, cy), radius, 0)
            cv2.circle(display_image, (int(cx), int(cy)), int(kp.size), (255, 0, 0), 1)

    print('done')
