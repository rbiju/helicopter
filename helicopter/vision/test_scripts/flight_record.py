import threading
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation
import pandas as pd
from tqdm import tqdm

from ultralytics import YOLO

from helicopter.utils import RemoteState, SymaRemoteRecorder, Profiler
from helicopter.vision import D435i
from helicopter.vision.point_detection import HelicopterYOLO, GPUImagePreprocessor, YOLOPointDetector
from helicopter.vision.tracking import TrianglePointMatcher

camera_quat = Rotation.from_rotvec(np.array([   0.008092,     0.25825,          -0]))
origin_quat = Rotation.from_rotvec(np.array([   -0.21642,     0.19777,     -1.7128]))
origin_position = np.array([     2.0493,    -0.16535,    -0.92545])
vertical_offset = np.array([          0,           0,    -0.12224])


def camera_to_table_space(_points: np.ndarray) -> np.ndarray:
    _points = _points - vertical_offset
    world_space_points = camera_quat.apply(_points)
    p_shifted = world_space_points - origin_position
    point_table = origin_quat.inv().apply(p_shifted)

    return point_table


class RemoteRecorderThread(threading.Thread):
    def __init__(self, _remote_state: RemoteState):
        super().__init__(daemon=True)
        self.remote_state = _remote_state
        self.running = True
        self.lock = threading.Lock()

    def run(self):
        while self.running:
            _commands = self.remote_state.recorder.read_command()

            if len(_commands) == 5:
                with self.lock:
                    self.remote_state.throttle = _commands[3]
                    self.remote_state.yaw = _commands[2]
                    self.remote_state.pitch = _commands[1]

            time.sleep(0.001)

    def convert_to_float(self):
        with self.lock:
            return self.remote_state.convert_to_float()

    def stop(self):
        self.running = False
        self.join()


if __name__ == '__main__':
    profiler = Profiler()
    imgsz = [720, 1280]
    camera = D435i(video_resolution=imgsz,
                   video_rate=30,
                   projector_power=0.,
                   autoexpose=False,
                   exposure_time=2400,
                   depth_preset=4)

    model = HelicopterYOLO(model=YOLO('/home/ray/yolo_models/helicopter/track_20260413_0/weights/best.engine',
                                      task='detect'),
                           preprocessor=GPUImagePreprocessor(imgsz=imgsz,
                                                             brightness_factor=1.0),
                           conf=0.1)
    detector = YOLOPointDetector(model=model,
                                 marker_tolerance=0.01,
                                 distance_threshold=4.0,
                                 marker_std_dev=0.01,
                                 margin=1)

    remote_state = RemoteState(recorder=SymaRemoteRecorder())
    rc_thread = RemoteRecorderThread(remote_state)

    try:
        camera.start()
        rc_thread.start()

        point_record = []
        commands = []
        first_video_time = None
        print("Starting Flight Recording")
        frame_count = 0

        while frame_count < 300:
            frames = camera.pipeline.wait_for_frames()
            video = camera.process_frames(frames)

            if first_video_time is None:
                first_video_time = video.depth_ts

            command = rc_thread.convert_to_float()
            commands.append((video.depth_ts - first_video_time, command))

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
                points, valid, invalid = detector.get_points_coords(video.depth_image,
                                                                    keypoints,
                                                                    camera.intrinsics)
                table_space_points = camera_to_table_space(points)
                point_record.append((video.depth_ts - first_video_time, table_space_points))
                profiler.end("Deproject")
                profiler.end("E2E")

    finally:
        rc_thread.stop()
        camera.stop()

    to_save = input('Save Flight Recording? (y/n) \n')
    if to_save.lower() == 'y':
        save_location = Path(__file__).parents[3] / 'notebooks' / 'flight_recordings' / 'recording.csv'
        point_matcher = TrianglePointMatcher(n=1000, k=50)

        df_dict = {'timestamp': [],
                   'command': [],
                   'position': [], }

        print('Computing flight states')
        for i in tqdm(range(len(commands))):
            timestamp = commands[i][0]
            command = commands[i][1]
            points = point_record[i][1]

            if len(points) < 3:
                continue
            else:
                profiler.start('Point_Matching')
                helicopter_state = point_matcher.get_alignment(points)
                profiler.end('Point_Matching')
                df_dict['timestamp'].append(timestamp)
                df_dict['command'].append(command)
                df_dict['position'].append(helicopter_state[1].tolist())

        print(profiler)

        df = pd.DataFrame(df_dict)
        df.to_csv(str(save_location), index=False)

    print('done')