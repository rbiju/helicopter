import cv2
from ultralytics import YOLO

from helicopter.utils import Profiler, Quitter, KeyListener
from helicopter.vision import D435i
from helicopter.vision.point_detection import HelicopterYOLO, GPUImagePreprocessor, YOLOPointDetector


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
                   exposure_time_rgb=1000,
                   depth_preset=3)

    model = HelicopterYOLO(model=YOLO('/home/ray/yolo_models/helicopter/track_20260413_0/weights/best.engine',
                                      task='detect'),
                           preprocessor=GPUImagePreprocessor(imgsz=imgsz,
                                                             brightness_factor=1.0),
                           conf=0.1)
    detector = YOLOPointDetector(model=model,
                                 marker_tolerance=0.01,
                                 distance_threshold=4.0,
                                 marker_std_dev=0.003,
                                 margin=1)
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
                keypoints = detector.get_refined_keypoints(video.ir_image, boxes)
                profiler.end("Keypoints")
                profiler.end("Detect")

                profiler.start("Deproject")
                points, valid, invalid = detector.get_points_coords(video.depth_image, keypoints, camera.intrinsics)
                profiler.end("Deproject")
                profiler.end("E2E")

                canvas = cv2.cvtColor(video.ir_image, cv2.COLOR_GRAY2RGB)

                if points is None:
                    detected_images.append(canvas)
                else:
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
