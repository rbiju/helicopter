import cv2
import numpy as np

from ultralytics import YOLO

from helicopter.utils import Profiler, Quitter, KeyListener
from helicopter.vision import D435i
from helicopter.vision.point_detection import HelicopterYOLO, GPUImagePreprocessor


def get_refined_keypoints(ir_frame, _boxes, margin=2):
    if len(_boxes) == 0:
        return []

    h, w = ir_frame.shape

    x1 = np.clip(_boxes[:, 0] - margin, 0, w).astype(int)
    y1 = np.clip(_boxes[:, 1] - margin, 0, h).astype(int)
    x2 = np.clip(_boxes[:, 2] + margin, 0, w).astype(int)
    y2 = np.clip(_boxes[:, 3] + margin, 0, h).astype(int)

    keypoints = []

    for i in range(len(_boxes)):
        roi = ir_frame[y1[i]:y2[i], x1[i]:x2[i]]

        if roi.size == 0:
            continue

        _, roi_clean = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        M = cv2.moments(roi_clean, binaryImage=False)

        if M["m00"] <= 0:
            continue

        cx = x1[i] + (M["m10"] / M["m00"])
        cy = y1[i] + (M["m01"] / M["m00"])

        _radius = np.sqrt(M["m00"] / 255.0 / np.pi)

        keypoints.append(cv2.KeyPoint(x=float(cx), y=float(cy), size=float(_radius)))

    return keypoints


def get_points_coords(depth_frame, keypoints, intrinsics):
    if not keypoints:
        return None

    h, w = depth_frame.shape

    valid_depths = []
    valid_uvs = []
    valid_radii = []

    valid_kps = []
    invalid_kps = []
    for kp in keypoints:
        cx, cy = kp.pt
        radius = kp.size / 2

        ix, iy = int(cx), int(cy)

        if ix < 0 or ix >= w or iy < 0 or iy >= h:
            invalid_kps.append(kp)
            continue

        safe_r = int(radius * 0.8)
        if safe_r < 1:
            safe_r = 1

        x0, x1 = max(0, ix - safe_r), min(w, ix + safe_r + 1)
        y0, y1 = max(0, iy - safe_r), min(h, iy + safe_r + 1)

        roi = depth_frame[y0:y1, x0:x1]

        valid_pixels = roi[roi > 0]

        if len(valid_pixels) < roi.size * 0.7:
            invalid_kps.append(kp)
            continue

        depth = np.mean(valid_pixels)
        d_std = np.std(valid_pixels)

        if d_std > 0.003:
            invalid_kps.append(kp)
            continue
            # depth = np.median(valid_pixels)

        if depth > 0.5 or depth <= 0:
            invalid_kps.append(kp)
            continue

        valid_depths.append(depth)
        valid_uvs.append((cx, cy))
        valid_radii.append(radius)

        valid_kps.append(kp)

    if not valid_depths:
        return None

    depths = np.array(valid_depths)
    uvs = np.array(valid_uvs)

    fx, fy = intrinsics.fx, intrinsics.fy
    ppx, ppy = intrinsics.ppx, intrinsics.ppy

    z_cam = depths
    x_cam = (uvs[:, 0] - ppx) * z_cam / fx
    y_cam = (uvs[:, 1] - ppy) * z_cam / fy

    final_points = np.column_stack((z_cam, -x_cam, -y_cam))

    return final_points, valid_kps, invalid_kps


if __name__ == '__main__':
    profiler = Profiler()
    camera = D435i(projector_power=360.,
                   autoexpose=False,
                   exposure_time=1800)
    model = HelicopterYOLO(model=YOLO('/home/ray/yolo_models/helicopter/measure_20260227/weights/best.engine',
                                      task='detect'),
                           preprocessor=GPUImagePreprocessor(),
                           conf=0.7)
    listener = KeyListener()
    quitter = Quitter(listener=listener)

    try:
        camera.start()
        detected_images = []
        frame_count = 0
        quitter.start()
        print("Starting detection")
        while frame_count < 500 and not quitter.quit:
            quitter.process()
            if quitter.quit:
                break

            frames = camera.pipeline.wait_for_frames()
            video = camera.process_frames(frames)
            frame_count += 1
            if video.ir_image is not None:
                profiler.start('E2E')
                profiler.start("Inference")
                profiler.start("Detect")

                boxes = model(video.ir_image)
                profiler.end("Inference")

                profiler.start('Keypoints')
                circles = get_refined_keypoints(video.ir_image, boxes, margin=2)
                profiler.end("Keypoints")
                profiler.end("Detect")

                profiler.start("Deproject")
                points = get_points_coords(video.depth_image, circles, camera.intrinsics)
                profiler.end("Deproject")
                profiler.end("E2E")

                if points is None:
                    continue
                else:
                    points, valid, invalid = points

                a = cv2.cvtColor(video.ir_image, cv2.COLOR_GRAY2RGB)
                a = cv2.drawKeypoints(video.ir_image, invalid, a, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
                a = cv2.drawKeypoints(a, valid, a, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
                detected_images.append(a)
    finally:
        camera.stop()
        quitter.stop()

    print(f"Frame count: {frame_count}")
    print(profiler)

    output_path = "detections.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 60.0, (640, 480), isColor=True)

    for frame in detected_images:
        out.write(frame)

    out.release()

    print('done')
