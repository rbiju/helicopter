import warnings
from collections import deque

import numpy as np
import pyrealsense2 as rs


class D435i:
    def __init__(self, accel_rate: int = 250,
                 gyro_rate: int = 200,
                 video_rate: int = 60,
                 video_resolution: tuple[int, int] = (640, 480),
                 enable_ir: bool = False,
                 enable_rgb: bool = False,
                 enable_motion: bool = False,
                 projector_power: float = 150.,
                 autoexpose: bool = True,
                 exposure_time: int = 800):
        self.ACCEL_RATE = accel_rate
        self.GYRO_RATE = gyro_rate

        # Depth is always enabled
        self.DEPTH_RESOLUTION = video_resolution
        self.DEPTH_RATE = video_rate
        self.IR_RESOLUTION = video_resolution
        self.IR_RATE = video_rate
        self.COLOR_RESOLUTION = video_resolution
        self.COLOR_RATE = video_rate

        serial = self.get_device_serial()

        self.enable_motion = enable_motion
        if enable_motion:
            self.imu_pipeline, self.imu_profile = self.setup_imu(serial)

        self.pipeline, self.profile, self.intrinsics = self.get_camera_pipeline(serial, enable_ir, enable_rgb)
        self.set_exposure(autoexpose, exposure_time)
        self.set_projector_power(projector_power)
        self.warmup_camera()

        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        self.hole_filler = rs.hole_filling_filter(1)

        self.align = rs.align(rs.stream.infrared)

        self.accel_queue = deque(maxlen=100)
        self.accel_time_queue = deque(maxlen=100)
        self.gyro_value = None
        self.gyro_time = 0.0

        self.last_imu_time = 0.0
        self.last_ir_time = 0.0

    def get_device_serial(self):
        ctx = rs.context()
        if not ctx.devices:
            raise RuntimeError("No RealSense device detected.")
        return ctx.devices[0].get_info(rs.camera_info.serial_number)

    def set_projector_power(self, power: float):
        if 0. <= power <= 360:
            print(f"Setting projector power to {power}")
            depth_device = self.profile.get_device()
            depth_sensor = depth_device.first_depth_sensor()
            depth_sensor.set_option(rs.option.laser_power, power)
            depth_sensor.set_option(rs.option.emitter_always_on, 1)
        else:
            warnings.warn(f"Specified projector power {power} is invalid. Must be between 0 and 360."
                          f"Projector power setting skipped.")

    def set_exposure(self, auto_exposure: bool, exposure: int):
        device = self.profile.get_device()
        depth_sensor = device.first_depth_sensor()

        if not auto_exposure:
            depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
            depth_sensor.set_option(rs.option.exposure, exposure)

            print(f"Auto-exposure disabled and exposure set to {exposure}")
        else:
            depth_sensor.set_option(rs.option.enable_auto_exposure, 1)

            print("Auto-exposure enabled")

    def get_camera_pipeline(self, serial, enable_ir: bool, enable_rgb: bool):
        ctx = rs.context()

        if len(ctx.devices) == 0:
            raise RuntimeError("No RealSense device detected.")

        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.depth, self.DEPTH_RESOLUTION[0], self.DEPTH_RESOLUTION[1], rs.format.z16,
                             self.DEPTH_RATE)

        if enable_ir:
            config.enable_stream(rs.stream.infrared, 1, self.IR_RESOLUTION[0], self.IR_RESOLUTION[1], rs.format.y8,
                                 self.IR_RATE)
        if enable_rgb:
            config.enable_stream(rs.stream.color, self.COLOR_RESOLUTION[0], self.COLOR_RESOLUTION[1], rs.format.rgb8,
                                 self.COLOR_RATE)

        pipeline = rs.pipeline(ctx)
        profile = pipeline.start(config)

        depth_stream = profile.get_stream(rs.stream.depth)
        depth_video_profile = depth_stream.as_video_stream_profile()
        depth_intrinsics = depth_video_profile.get_intrinsics()

        print("Successfully retrieved Depth Intrinsics:")
        print(f"  Resolution: {depth_intrinsics.width}x{depth_intrinsics.height}")
        print(f"  Focal Length (fx, fy): ({depth_intrinsics.fx:.2f}, {depth_intrinsics.fy:.2f})")
        print(f"  Principal Point (cx, cy): ({depth_intrinsics.ppx:.2f}, {depth_intrinsics.ppy:.2f})")

        return pipeline, profile, depth_intrinsics

    def warmup_camera(self):
        print("Warming up camera... waiting 200 frames.")
        for _ in range(200):
            self.pipeline.wait_for_frames()

    def setup_imu(self, serial):
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, self.ACCEL_RATE)
        config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, self.GYRO_RATE)

        pipeline = rs.pipeline()
        profile = pipeline.start(config)

        device = profile.get_device()
        motion_sensor = device.first_motion_sensor()
        motion_sensor.set_option(rs.option.global_time_enabled, 0)

        print("Warming up imu... waiting 100 frames.")
        for _ in range(100):
            pipeline.wait_for_frames()

        return pipeline, profile

    def process_frames(self, frames: rs.composite_frame):
        frames = self.align.process(frames)

        depth_frame = frames.get_depth_frame()
        depth_frame = self.hole_filler.process(depth_frame)
        ir_frame = frames.get_infrared_frame(1)

        if depth_frame:
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_image = depth_image * self.depth_scale
            ts_depth = depth_frame.get_timestamp()
        else:
            depth_image = None
            ts_depth = None
        if ir_frame:
            ir_image = np.asanyarray(ir_frame.get_data())
            ts_ir = ir_frame.get_timestamp()
        else:
            ir_image = None
            ts_ir = None

        return depth_image, ts_depth, ir_image, ts_ir

    def process_imu_frames(self, frames):
        accel_data = None
        ts_accel = None
        gyro_data = None
        ts_gyro = None

        for frame in frames:
            if frame.get_profile().stream_type() == rs.stream.accel:
                accel_data = frame.as_motion_frame().get_motion_data()
                # coordinate transform so +Z is pointing up and +X is pointing out of the camera
                accel_data = np.array([accel_data.z, -accel_data.x, -accel_data.y])
                ts_accel = frame.get_timestamp() / 1000.
            elif frame.get_profile().stream_type() == rs.stream.gyro:
                gyro_data = frame.as_motion_frame().get_motion_data()
                gyro_data = np.array([gyro_data.z, -gyro_data.x, -gyro_data.y])
                ts_gyro = frame.get_timestamp() / 1000.

        return accel_data, ts_accel, gyro_data, ts_gyro

    def get_synced_imu(self, imu_frames):
        accel_data, ts_accel, gyro_data, ts_gyro = self.process_imu_frames(imu_frames)
        if accel_data is not None:
            self.accel_queue.append(accel_data)
            self.accel_time_queue.append(ts_accel)
            if len(self.accel_queue) > 50:
                accel_x = np.interp(self.gyro_time, np.array(list(self.accel_time_queue))[:2],
                                    np.array(list(self.accel_queue))[:2, 0])
                accel_y = np.interp(self.gyro_time, np.array(list(self.accel_time_queue))[:2],
                                    np.array(list(self.accel_queue))[:2, 1])
                accel_z = np.interp(self.gyro_time, np.array(list(self.accel_time_queue))[:2],
                                    np.array(list(self.accel_queue))[:2, 2])
                accel_data = np.array([accel_x, accel_y, accel_z])

                return accel_data, self.gyro_value, self.gyro_time

        if gyro_data is not None:
            self.last_imu_time = self.gyro_time
            self.gyro_time = ts_gyro
            self.gyro_value = gyro_data

        return None

    def stop(self):
        print("Closing D435i pipelines")
        if self.enable_motion:
            self.imu_pipeline.stop()
        self.pipeline.stop()
