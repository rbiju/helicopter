import warnings
from collections import deque

import numpy as np
import pyrealsense2 as rs


class D435i:
    def __init__(self, accel_rate: int = 250,
                 gyro_rate: int = 200,
                 video_rate: int = 60,
                 video_resolution: tuple[int, int] = (640, 480),
                 enable_depth: bool = False,
                 enable_rgb: bool = False,
                 enable_motion: bool = False,
                 projector_power: float = 150.,
                 autoexpose: bool = True,
                 exposure_time: int = 800):
        self.ACCEL_RATE = accel_rate
        self.GYRO_RATE = gyro_rate

        # IR is always enabled
        self.IR_RESOLUTION = video_resolution
        self.IR_RATE = video_rate
        self.DEPTH_RESOLUTION = video_resolution
        self.DEPTH_RATE = video_rate
        self.COLOR_RESOLUTION = video_resolution
        self.COLOR_RATE = video_rate

        serial = self.get_device_serial()

        self.enable_motion = enable_motion
        if enable_motion:
            self.imu_pipeline, self.imu_profile = self.setup_imu(serial)

        self.pipeline, self.profile, self.intrinsics = self.get_camera_pipeline(serial, enable_depth, enable_rgb)
        depth_sensor = self.profile.get_device().first_depth_sensor()
        depth_sensor.set_option(rs.option.visual_preset, 3)
        self.set_exposure(autoexpose, exposure_time)
        self.set_projector_power(projector_power)
        self.warmup_camera()

        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        self.hdr_merge = rs.hdr_merge()
        self.temporal_filter = rs.temporal_filter(smooth_alpha=0.40, smooth_delta=20.0, persistence_control=3)

        self.align = rs.align(rs.stream.infrared)

        self.gyro_queue = deque(maxlen=50)
        self.accel_time_queue = deque(maxlen=50)
        self.gyro_time_queue = deque(maxlen=50)

    @property
    def last_imu_time(self):
        return self.accel_time_queue[-2]

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

    def get_camera_pipeline(self, serial, enable_depth: bool, enable_rgb: bool):
        ctx = rs.context()

        if len(ctx.devices) == 0:
            raise RuntimeError("No RealSense device detected.")

        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.infrared, 1, self.IR_RESOLUTION[0], self.IR_RESOLUTION[1], rs.format.y8,
                             self.IR_RATE)
        if enable_depth:
            config.enable_stream(rs.stream.depth, self.DEPTH_RESOLUTION[0], self.DEPTH_RESOLUTION[1], rs.format.z16,
                                 self.DEPTH_RATE)
        if enable_rgb:
            config.enable_stream(rs.stream.color, self.COLOR_RESOLUTION[0], self.COLOR_RESOLUTION[1], rs.format.rgb8,
                                 self.COLOR_RATE)

        pipeline = rs.pipeline(ctx)
        profile = pipeline.start(config)

        ir_stream = profile.get_stream(rs.stream.infrared, 1)
        ir_video_profile = ir_stream.as_video_stream_profile()
        ir_intrinsics = ir_video_profile.get_intrinsics()

        print("Successfully retrieved IR Intrinsics:")
        print(f"  Resolution: {ir_intrinsics.width}x{ir_intrinsics.height}")
        print(f"  Focal Length (fx, fy): ({ir_intrinsics.fx:.2f}, {ir_intrinsics.fy:.2f})")
        print(f"  Principal Point (cx, cy): ({ir_intrinsics.ppx:.2f}, {ir_intrinsics.ppy:.2f})")

        return pipeline, profile, ir_intrinsics

    def warmup_camera(self):
        print("Warming up camera... waiting 60 frames.")
        for _ in range(60):
            self.pipeline.wait_for_frames()

    def setup_imu(self, serial):
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, self.ACCEL_RATE)
        config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, self.GYRO_RATE)

        pipeline = rs.pipeline()
        profile = pipeline.start(config)

        # device = profile.get_device()
        # motion_sensor = device.first_motion_sensor()
        # motion_sensor.set_option(rs.option.global_time_enabled, 0)

        print("Warming up imu... waiting 100 frames.")
        for _ in range(100):
            pipeline.wait_for_frames()

        return pipeline, profile

    def process_frames(self, frames: rs.composite_frame):
        frames = self.align.process(frames)

        depth_frame = frames.get_depth_frame()
        depth_frame = self.hdr_merge.process(depth_frame)
        depth_frame = self.temporal_filter.process(depth_frame)
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
        # rs returns gyro and most recent accelerometer data in the same packet
        accel_data, ts_accel, gyro_data, ts_gyro = self.process_imu_frames(imu_frames)
        if accel_data is not None and gyro_data is not None:
            if len(self.accel_time_queue) > 0:
                if ts_accel == self.accel_time_queue[-1]:
                    return None
                if ts_gyro == self.gyro_time_queue[-1]:
                    return None
            self.gyro_queue.append(gyro_data)
            self.accel_time_queue.append(ts_accel)
            self.gyro_time_queue.append(ts_gyro)
            if len(self.gyro_queue) > 20:
                gyro_x = np.interp(ts_accel, np.array(list(self.accel_time_queue))[-2:],
                                   np.array(list(self.gyro_queue))[-2:, 0])
                gyro_y = np.interp(ts_accel, np.array(list(self.accel_time_queue))[-2:],
                                   np.array(list(self.gyro_queue))[-2:, 1])
                gyro_z = np.interp(ts_accel, np.array(list(self.accel_time_queue))[-2:],
                                   np.array(list(self.gyro_queue))[-2:, 2])
                gyro_interpolated = np.array([gyro_x, gyro_y, gyro_z])

                return accel_data, gyro_interpolated, ts_accel

            else:
                return None

        return None

    def stop(self):
        print("Closing D435i pipelines")
        if self.enable_motion:
            self.imu_pipeline.stop()
        self.pipeline.stop()
