import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from helicopter.utils import Profiler, Quitter, KeyListener
from helicopter.vision import D435i


def get_aruco_poses(image, intrinsics, marker_size_meters, _detector):
    _corners, _ids, rejected = _detector.detectMarkers(image)

    if _ids is None:
        return None, None, None, None, None, None

    camera_matrix = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float64)

    half_size = marker_size_meters / 2.0
    obj_points = np.array([
        [-half_size, half_size, 0],
        [half_size, half_size, 0],
        [half_size, -half_size, 0],
        [-half_size, -half_size, 0]
    ], dtype=np.float32)

    rvecs_out = []
    tvecs_out = []

    for corner in _corners:
        img_points = corner[0].astype(np.float32)
        success, _rvecs, _tvecs, _ = cv2.solvePnPGeneric(
            obj_points,
            img_points,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )

        if success and len(_rvecs) > 0:
            best_rvec = _rvecs[0]
            best_tvec = _tvecs[0]

            if len(_rvecs) > 1:
                R0, _ = cv2.Rodrigues(_rvecs[0])

                flip_180 = np.array([
                    [1, 0, 0],
                    [0, -1, 0],
                    [0, 0, -1]
                ], dtype=np.float64)

                R_offset = R0 @ flip_180

                best_rvec, _ = cv2.Rodrigues(R_offset)

            rvecs_out.append(best_rvec)
            tvecs_out.append(best_tvec)
        else:
            rvecs_out.append(None)
            tvecs_out.append(None)

    return _corners, _ids, rvecs_out, tvecs_out, camera_matrix, dist_coeffs


if __name__ == '__main__':
    profiler = Profiler()
    imgsz = [720, 1280]

    camera = D435i(video_resolution=imgsz,
                   video_rate=30,
                   enable_rgb=True,
                   projector_power=0.,
                   autoexpose=False,
                   exposure_time=3600,
                   autoexpose_rgb=False,
                   exposure_time_rgb=300)

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.cornerRefinementWinSize = 5
    parameters.cornerRefinementMaxIterations = 100
    parameters.cornerRefinementMinAccuracy = 0.001
    parameters.minMarkerPerimeterRate = 0.01
    parameters.minMarkerDistanceRate = 0.05
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 53
    parameters.adaptiveThreshWinSizeStep = 5
    parameters.adaptiveThreshConstant = 5
    parameters.polygonalApproxAccuracyRate = 0.03
    parameters.errorCorrectionRate = 0.6

    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
    MARKER_SIZE_METERS = 0.0427

    listener = KeyListener()
    quitter = Quitter(listener=listener)

    try:
        camera.start()
        detected_images = []
        frame_count = 0
        marker_dict = {}
        quitter.start()
        print("Starting ArUco detection")

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

                corners, ids, rvecs, tvecs, cam_mat, dist = get_aruco_poses(
                    video.color_image,
                    camera.intrinsics,
                    MARKER_SIZE_METERS,
                    detector
                )

                profiler.end("Detect")

                profiler.start("Draw")
                canvas = cv2.cvtColor(video.color_image, cv2.COLOR_RGB2BGR)

                if ids is not None:
                    cv2.aruco.drawDetectedMarkers(canvas, corners, ids)

                    axis_length = MARKER_SIZE_METERS * 0.75
                    for idx, rvec, tvec in zip(ids, rvecs, tvecs):
                        if rvec is not None and tvec is not None:
                            cv2.drawFrameAxes(
                                canvas,
                                cam_mat,
                                dist,
                                rvec,
                                tvec,
                                axis_length
                            )

                            if int(idx[0]) not in marker_dict:
                                marker_dict[int(idx[0])] = {'position': tvec.flatten(),
                                                            'orientation': Rotation.from_rotvec(rvec.flatten())
                                                            .as_rotvec(degrees=True)}


                profiler.end("Draw")
                profiler.end("E2E")

                detected_images.append(canvas)

    finally:
        camera.stop()
        quitter.stop()

    print(profiler)
    print(marker_dict)

    output_path = "aruco_tracking.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (imgsz[1], imgsz[0]), isColor=True)

    for frame in detected_images:
        out.write(frame)

    out.release()

    print('done')