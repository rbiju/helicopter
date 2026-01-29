import time

import cv2

from ultralytics import YOLO

from helicopter.vision import D435i
from helicopter.vision.point_detection import HelicopterYOLO


def draw_subpixel_circle(center, _radius, shift=4):
    factor = 1 << shift

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
    frame_count = 0
    while frame_count < 1:
        frames = camera.pipeline.wait_for_frames()
        depth_image, ts_depth, ir_image, ts_ir, laser_state = camera.process_frames(frames)
        frame_count += 1

    camera.stop()

    model = HelicopterYOLO(model=YOLO('/home/ray/yolo_models/helicopter/measure/weights/best.engine',
                                      task='detect'),
                           conf=0.75)

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

            circles.append(((final_x, final_y), radius))

        end_detect = time.perf_counter()
        detect_time = end_detect - start_detect

        display_image = ir_image.copy()
        display_image = cv2.cvtColor(display_image, cv2.COLOR_GRAY2BGR)
        for circle in circles:
            (cx, cy), radius = circle
            c_sub, r_sub = draw_subpixel_circle((cx, cy), radius, 0)
            cv2.circle(display_image, c_sub, r_sub, (0, 255, 0), 1, shift=0)

    print('done')
