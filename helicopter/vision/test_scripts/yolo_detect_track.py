import cv2

from ultralytics import YOLO

from helicopter.utils import Profiler, Quitter, KeyListener
from helicopter.vision import D435i
from helicopter.vision.point_detection import HelicopterYOLO, GPUImagePreprocessor

from .yolo_detect import get_points_coords, get_refined_keypoints


if __name__ == '__main__':
    profiler = Profiler()
    camera = D435i(video_resolution=[720, 1280],
                   video_rate=30,
                   projector_power=0.,
                   autoexpose=False,
                   exposure_time=2000)

    model = HelicopterYOLO(model=YOLO('/home/ray/yolo_models/helicopter/track_20260320_0/weights/best.engine',
                                      task='detect'),
                           preprocessor=GPUImagePreprocessor(),
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
            depth_image, ts_depth, ir_image, ts_ir, laser_state = camera.process_frames(frames)
            frame_count += 1
            if ir_image is not None:
                profiler.start('E2E')
                profiler.start("Inference")
                profiler.start("Detect")

                boxes = model(ir_image)
                profiler.end("Inference")

                profiler.start('Keypoints')
                circles = get_refined_keypoints(ir_image, boxes, margin=2)
                profiler.end("Keypoints")
                profiler.end("Detect")

                profiler.start("Deproject")
                points = get_points_coords(depth_image, circles, camera.intrinsics)
                profiler.end("Deproject")
                profiler.end("E2E")

                if points is None:
                    continue
                else:
                    points, valid, invalid = points

                a = cv2.cvtColor(ir_image, cv2.COLOR_GRAY2RGB)
                a = cv2.drawKeypoints(ir_image, invalid, a, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
                a = cv2.drawKeypoints(a, valid, a, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
                detected_images.append(a)
    finally:
        camera.stop()
        quitter.stop()

    print(f"Frame count: {frame_count}")
    print(profiler)

    output_path = "detections_tracking.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 60.0, (1280, 720), isColor=True)

    for frame in detected_images:
        out.write(frame)

    out.release()

    print('done')
