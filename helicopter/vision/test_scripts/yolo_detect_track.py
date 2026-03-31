import cv2
import numpy as np
from ultralytics import YOLO

from helicopter.utils import Profiler, Quitter, KeyListener
from helicopter.vision import D435i
from helicopter.vision.point_detection import HelicopterYOLO, GPUImagePreprocessor

from helicopter.vision.test_scripts.yolo_detect import get_refined_keypoints


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

        if d_std > 0.01:
            invalid_kps.append(kp)
            continue

        if depth <= 0:
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
    imgsz = [720, 1280]
    camera = D435i(video_resolution=imgsz,
                   video_rate=30,
                   enable_rgb=True,
                   projector_power=0.,
                   autoexpose=False,
                   exposure_time=2400,
                   autoexpose_rgb=False,
                   exposure_time_rgb=2400)

    model = HelicopterYOLO(model=YOLO('/home/ray/yolo_models/helicopter/track_20260327_0/weights/best.engine',
                                      task='detect'),
                           preprocessor=GPUImagePreprocessor(imgsz=imgsz),
                           imgsz=imgsz,
                           conf=0.1)
    listener = KeyListener()
    quitter = Quitter(listener=listener)

    try:
        camera.start()
        detected_images = []
        frame_count = 0
        quitter.start()
        print("Starting detection")
        while frame_count < 100 and not quitter.quit:
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

                canvas = cv2.cvtColor(video.ir_image, cv2.COLOR_GRAY2RGB)

                if points is None:
                    detected_images.append(canvas)
                else:
                    coords, valid, invalid = points
                    canvas = cv2.drawKeypoints(canvas, invalid, None,
                                               color=(0, 0, 255),
                                               flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
                    canvas = cv2.drawKeypoints(canvas, valid, None,
                                               color=(0, 255, 0),
                                               flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)

                    detected_images.append(canvas)

    finally:
        camera.stop()
        quitter.stop()

    print(profiler)

    output_path = "detections_tracking.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (1280, 720), isColor=True)

    for frame in detected_images:
        out.write(frame)

    out.release()

    print('done')
