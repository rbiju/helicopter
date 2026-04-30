from functools import partial
from pathlib import Path
from queue import Queue
from threading import Event, Lock
import time

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from helicopter.aircraft.base import Aircraft
from helicopter.configuration import HydraConfigurable, LocalHydraConfiguration
from helicopter.remote import RemoteRecorderThread, SymaRemoteRecorder
from helicopter.vision.tracking import Tracker, TrianglePointMatcher
from helicopter.utils import Profiler

from .base import Task


class FakeSharedMemory:
    def __init__(self, size: int):
        self.buf = bytearray(size)
    def close(self):
        pass
    def unlink(self):
        pass


@HydraConfigurable
class RecordFlight(Task):
    def __init__(self, filename: str = 'recording'):
        super().__init__()

        aircraft_dummy = np.zeros(shape=(Aircraft.N,), dtype=np.float64)
        aircraft_sm = FakeSharedMemory(size=aircraft_dummy.nbytes)
        kill_event = Event()

        config = LocalHydraConfiguration('/home/ray/projects/helicopter/configs/atomic/tracker.yaml')
        tracker: partial = Tracker.from_hydra_configuration(config)
        self.tracker: Tracker = tracker(aircraft_sm=aircraft_sm,
                                        kill_signal=kill_event, )

        self.profiler = Profiler()
        self.filename = filename + '.csv'

    def run(self, **kwargs):
        marker_queue = Queue()
        origin_queue = Queue()
        orientation_ready = Event()
        origin_queue.put([{'id': 0,
                           'position': np.array([-0.355 + 0.035, -0.685 + 0.035, 0]),
                           'rotation': Rotation.from_euler('y', [90], degrees=True)},
                          {'id': 1,
                           'position': np.array([-0.355, -0.685 - 0.035, -0.035]),
                           'rotation': Rotation.from_euler('z', [90], degrees=True)},
                          {'id': 2,
                           'position': np.array([-0.355, -0.685, -0.035]),
                           'rotation': Rotation.from_euler('z', [0], degrees=True)},
                          ])

        lock = Lock()

        self.tracker.initialize(marker_queue=marker_queue,
                           origin_queue=origin_queue,
                           orientation_ready=orientation_ready,
                           aircraft_lock=lock, )

        rc_thread = RemoteRecorderThread(SymaRemoteRecorder())

        rc_thread.start()
        point_record = []
        commands = []
        first_video_time = None
        print("Starting Flight Recording")
        frame_count = 0

        try:
            while frame_count < 300:
                frames = self.tracker.camera.pipeline.wait_for_frames()
                video = self.tracker.camera.process_frames(frames)

                if first_video_time is None:
                    first_video_time = video.ir_ts

                frame_count += 1
                if video.ir_image is not None:
                    self.profiler.start('E2E')

                    self.profiler.start("Inference")
                    keypoints = self.tracker.point_handler.detector.detect(video.ir_image)
                    self.profiler.end("Inference")
                    self.profiler.start("Deproject")
                    points, _, _ = (
                        self.tracker.point_handler.detector.get_points_coords(video.depth_image,
                                                                              keypoints,
                                                                              self.tracker.camera.intrinsics))
                    self.profiler.end("Deproject")

                    table_space_points = self.tracker.camera_to_table_space(points)
                    point_record.append((video.ir_ts - first_video_time, table_space_points))
                    command = rc_thread.get_commands().tolist()
                    commands.append((video.ir_ts - first_video_time, command))

                    self.profiler.end("E2E")
                    time.sleep(0.00001)
        finally:
            self.tracker.cleanup()
            rc_thread.stop()

        to_save = input('Save Flight Recording? (y/n) \n')
        if to_save.lower() == 'y':
            save_location = Path(__file__).parents[2] / 'notebooks' / 'flight_recordings' / self.filename
            point_matcher = TrianglePointMatcher(n=2500, k=100)

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
                    self.profiler.start('Point_Matching')
                    helicopter_state = point_matcher.get_alignment(points)
                    self.profiler.end('Point_Matching')
                    df_dict['timestamp'].append(timestamp)
                    df_dict['command'].append(command)
                    df_dict['position'].append(helicopter_state[1].tolist())

            print(self.profiler)

            df = pd.DataFrame(df_dict)
            df.to_csv(str(save_location), index=False)
