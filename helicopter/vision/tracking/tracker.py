import queue
import threading
import time

from helicopter.aircraft import Aircraft, AircraftManager
from helicopter.vision import D435i
from .point_handler import TrackingPointHandler


class Tracker:
    def __init__(self, aircraft: AircraftManager, point_handler: TrackingPointHandler, camera: D435i):
        self.aircraft_manager = aircraft
        self.point_handler = point_handler

        self.camera = camera

        self.vision_thread = threading.Thread(target=self.vision_loop, daemon=True)
        self.vision_queue = queue.Queue(maxsize=5)

        self.initialized = False
        self.is_running = False

    def vision_loop(self):
        while self.is_running:
            frames = self.camera.pipeline.wait_for_frames()

            depth_image, ts_depth, ir_image, ts_ir, laser_state = self.camera.process_frames(frames)

            if ir_image is not None:
                try:
                    self.vision_queue.put((ts_depth, ir_image, depth_image), block=False)
                except queue.Full:
                    self.vision_queue.get_nowait()
                    self.vision_queue.put((ts_depth, ir_image, depth_image), block=False)


    def initialize_orientation(self):
        with self.aircraft_manager as manager:
            aircraft: Aircraft = manager.Aircraft()
            aircraft.set_flight_state('Init')

            orientation_iters = 200
            print("Initializing helicopter orientation. Do not move aircraft.")
            counter = 0
            bad_frames = 0
            while counter < orientation_iters:
                if bad_frames > 50:
                    raise RuntimeError('Helicopter points not visible in 25% of init frames. Check that aircraft is in view.')
                frames = self.camera.pipeline.wait_for_frames()
                depth_image, ts_depth, ir_image, ts_ir, laser_state = self.camera.process_frames(frames)
                measure_out = self.point_handler.get_measured_points(ir_frame=ir_image,
                                                                     depth_frame=depth_image,
                                                                     intrinsics=self.camera.intrinsics)
                if measure_out is not None:
                    marker_coords, keypoints = measure_out
                    self.point_handler.register_points(marker_coords)
                else:
                    bad_frames += 1
                    continue
                counter += 1

            r, t = self.point_handler.matcher.get_alignment(self.point_handler.init_points_coords)

            aircraft.set_quaternion(r)
            aircraft.set_position(t)

            self.initialized = True

    def track(self):
        self.is_running = True
        self.vision_thread.start()

        while self.is_running:
            try:
                vision_time, ir_frame, depth_frame = self.vision_queue.get(timeout=0.05)
                vision_time = float(vision_time)
            except queue.Empty:
                time.sleep(0.001)
                continue

            # Perform UKF Prediction here

            measured_out = self.point_handler.get_measured_points(ir_frame=ir_frame, depth_frame=depth_frame, intrinsics=self.camera.intrinsics)
            if measured_out is not None:
                marker_coords, keypoints = measured_out

                # Perform ICP + UKF Update here
            else:
                continue

