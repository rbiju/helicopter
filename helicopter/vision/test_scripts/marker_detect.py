import cv2
import numpy as np

from helicopter.utils import Profiler, Quitter, KeyListener
from helicopter.vision.point_detection import AprilTagMarkerDetector
from helicopter.vision import D435i

COB_MATRIX = np.array([
    [0, 0, 1],
    [-1, 0, 0],
    [0, -1, 0]
])


if __name__ == '__main__':
    profiler = Profiler()
    imgsz = [720, 1280]

    camera = D435i(video_resolution=imgsz,
                   video_rate=30,
                   enable_rgb=True,
                   projector_power=0.,
                   autoexpose=False,
                   exposure_time=3200,
                   autoexpose_rgb=False,
                   exposure_time_rgb=500)

    detector = AprilTagMarkerDetector()
    detector.activate(camera.color_intrinsics, camera.color_ir_extrinsics)

    listener = KeyListener()
    quitter = Quitter(listener=listener)

    try:
        camera.start()
        detected_images = []
        frame_count = 0
        marker_dict = {}
        quitter.start()
        print("Starting Marker detection")

        while frame_count < 100 and not quitter.quit:
            quitter.process()
            if quitter.quit:
                break

            frames = camera.pipeline.wait_for_frames()
            video = camera.process_frames(frames)
            frame_count += 1

            if video.color_image is not None:
                profiler.start('E2E')
                profiler.start("Detect")

                detected_markers = detector.detect_markers(video.color_image)

                profiler.end("Detect")

                profiler.start("Draw")
                canvas = cv2.cvtColor(video.color_image, cv2.COLOR_RGB2BGR)

                if len(detected_markers) > 0:
                    axis_length = detector.marker_size_meters * 0.75
                    for marker in detected_markers:
                        rvec = marker.unaligned_rotation.as_rotvec()
                        tvec = np.array(marker.unaligned_position, dtype=np.float32).reshape(3, 1)
                        cv2.drawFrameAxes(
                            canvas,
                            detector.intrinsic_matrix,
                            detector.dist_coeffs,
                            rvec,
                            tvec,
                            axis_length
                        )

                        if marker.id not in marker_dict:
                            marker_dict[marker.id] = {'position': marker.position,
                                                      'orientation': marker.rotation.as_rotvec(degrees=True)}

                profiler.end("Draw")
                profiler.end("E2E")

                detected_images.append(canvas)

    finally:
        camera.stop()
        quitter.stop()

    print(profiler)
    print(marker_dict)

    output_path = "apriltag_tracking.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (imgsz[1], imgsz[0]), isColor=True)

    for frame in detected_images:
        out.write(frame)

    out.release()

    print('done')