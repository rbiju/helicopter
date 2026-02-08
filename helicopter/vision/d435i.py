import warnings
from collections import deque

import numpy as np
import pyrealsense2 as rs


class D435i:
    def __init__(self, accel_rate: int = 250,
                 gyro_rate: int = 200,
                 video_rate: int = 60,
                 video_resolution: tuple[int, int] = (480, 640),
                 enable_rgb: bool = False,
                 enable_motion: bool = False,
                 projector_power: float = 150.,
                 toggle_projector: bool = False,
                 autoexpose: bool = True,
                 exposure_time: int = 800,
                 ema_factor: float = 1.0):
        self.ACCEL_RATE = accel_rate
        self.GYRO_RATE = gyro_rate

        # IR is always enabled
        self.IR_RESOLUTION = video_resolution
        self.IR_RATE = video_rate
        self.DEPTH_RESOLUTION = video_resolution
        self.DEPTH_RATE = video_rate
        self.COLOR_RESOLUTION = video_resolution
        self.COLOR_RATE = video_rate

        # Sensor setup
        serial = self.get_device_serial()
        ctx = rs.context()
        device = self.get_device_from_serial(ctx, serial)
        depth_sensor = device.first_depth_sensor()
        depth_sensor.set_option(rs.option.visual_preset, 4)
        self.depth_scale = depth_sensor.get_depth_scale()
        self.set_exposure(depth_sensor, autoexposure=autoexpose, exposure_time=exposure_time)
        self.set_projector_power(depth_sensor, projector_power)

        if toggle_projector:
            print("Laser projector toggle enabled")
            depth_sensor.set_option(rs.option.emitter_always_on, 0)
            depth_sensor.set_option(rs.option.emitter_on_off, 1)
        else:
            print('Emitter always on enabled')
            depth_sensor.set_option(rs.option.emitter_always_on, 1)

        depth_sensor.set_option(rs.option.frames_queue_size, 1)

        self.enable_motion = enable_motion
        if enable_motion:
            self.imu_pipeline, self.imu_config = self.get_imu_pipeline(serial)

        self.pipeline, self.config, self.intrinsics = self.get_camera_pipeline(serial, enable_rgb)

        self.hdr_merge = rs.hdr_merge()

        self.align = rs.align(rs.stream.infrared)

        self.ema_factor = ema_factor
        self.last_accel = None
        self.last_gyro = None

        self.time_queue = deque(maxlen=10)

    @property
    def last_imu_time(self):
        return self.time_queue[-2]

    def get_device_from_serial(self, ctx, serial):
        for dev in ctx.devices:
            if dev.get_info(rs.camera_info.serial_number) == serial:
                return dev
        raise RuntimeError(f"Device {serial} not found")

    def get_device_serial(self):
        ctx = rs.context()
        if not ctx.devices:
            raise RuntimeError("No RealSense device detected.")
        return ctx.devices[0].get_info(rs.camera_info.serial_number)

    def set_projector_power(self, depth_sensor: rs.depth_sensor, power: float):
        if 0. <= power <= 360:
            print(f"Setting projector power to {power}")
            depth_sensor.set_option(rs.option.laser_power, power)
            depth_sensor.set_option(rs.option.emitter_enabled, 1)
        else:
            warnings.warn(f"Specified projector power {power} is invalid. Must be between 0 and 360."
                          f"Projector power setting skipped.")

    def set_exposure(self, depth_sensor: rs.depth_sensor, autoexposure: bool, exposure_time: int):
        if not autoexposure:
            depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
            depth_sensor.set_option(rs.option.exposure, exposure_time)

            print(f"Auto-exposure disabled and exposure set to {exposure_time}")
        else:
            depth_sensor.set_option(rs.option.enable_auto_exposure, 1)

            print("Auto-exposure enabled")

    @staticmethod
    def get_intrinsics(serial, stream_type=rs.stream.infrared, stream_index=1, width=640, height=480, fps=30,
                       _format=rs.format.y8):
        ctx = rs.context()
        devices = [dev for dev in ctx.devices if dev.get_info(rs.camera_info.serial_number) == serial]
        if not devices:
            raise RuntimeError(f"Device {serial} not found")
        device = devices[0]

        for sensor in device.query_sensors():
            for profile in sensor.get_stream_profiles():
                if (profile.stream_type() == stream_type and
                        profile.stream_index() == stream_index and
                        profile.format() == _format and
                        profile.fps() == fps):

                    if profile.is_video_stream_profile():
                        v_profile = profile.as_video_stream_profile()
                        if v_profile.width() == width and v_profile.height() == height:
                            return v_profile.get_intrinsics()

        raise RuntimeError(f"Stream configuration not found for device {serial}")

    def get_camera_pipeline(self, serial, enable_rgb: bool):
        ctx = rs.context()

        if len(ctx.devices) == 0:
            raise RuntimeError("No RealSense device detected.")

        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.infrared, 1, self.IR_RESOLUTION[1], self.IR_RESOLUTION[0], rs.format.y8,
                             self.IR_RATE)
        config.enable_stream(rs.stream.depth, self.DEPTH_RESOLUTION[1], self.DEPTH_RESOLUTION[0], rs.format.z16,
                             self.DEPTH_RATE)
        if enable_rgb:
            config.enable_stream(rs.stream.color, self.COLOR_RESOLUTION[1], self.COLOR_RESOLUTION[0], rs.format.rgb8,
                                 self.COLOR_RATE)

        pipeline = rs.pipeline(ctx)

        ir_intrinsics = self.get_intrinsics(serial,
                                            rs.stream.infrared,
                                            1,
                                            self.IR_RESOLUTION[1],
                                            self.IR_RESOLUTION[0],
                                            self.IR_RATE, rs.format.y8)

        print("Successfully retrieved IR Intrinsics:")
        print(f"  Resolution: {ir_intrinsics.width}x{ir_intrinsics.height}")
        print(f"  Focal Length (fx, fy): ({ir_intrinsics.fx:.2f}, {ir_intrinsics.fy:.2f})")
        print(f"  Principal Point (cx, cy): ({ir_intrinsics.ppx:.2f}, {ir_intrinsics.ppy:.2f})")

        return pipeline, config, ir_intrinsics

    def get_imu_pipeline(self, serial):
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, self.ACCEL_RATE)
        config.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, self.GYRO_RATE)

        pipeline = rs.pipeline()

        ctx = rs.context()
        device = ctx.query_devices()[0]
        motion_sensor = device.first_motion_sensor()

        motion_sensor.set_option(rs.option.frames_queue_size, 1)

        return pipeline, config

    def start(self):
        if self.enable_motion:
            self.imu_pipeline.start(self.imu_config)

            print("Warming up imu... waiting 100 frames.")
            for _ in range(100):
                self.imu_pipeline.wait_for_frames()

        self.pipeline.start(self.config)

        print("Warming up camera... waiting 60 frames.")
        for _ in range(60):
            self.pipeline.wait_for_frames()

    def process_frames(self, frames: rs.composite_frame):
        frames = self.align.process(frames)

        depth_frame = frames.get_depth_frame()
        laser_state = (depth_frame.get_frame_metadata(rs.frame_metadata_value.frame_laser_power_mode) == 1)
        depth_frame = self.hdr_merge.process(depth_frame)
        ir_frame = frames.get_infrared_frame(1)

        if depth_frame:
            depth_image = np.asanyarray(depth_frame.get_data()).copy()
            depth_image = depth_image * self.depth_scale
            ts_depth = depth_frame.get_timestamp() / 1000.
        else:
            depth_image = None
            ts_depth = None
        if ir_frame:
            ir_image = np.asanyarray(ir_frame.get_data()).copy()
            ts_ir = ir_frame.get_timestamp() / 1000.
        else:
            ir_image = None
            ts_ir = None

        return depth_image, ts_depth, ir_image, ts_ir, laser_state

    def process_imu_frames(self, frames):
        accel_data = None
        ts_accel = None
        gyro_data = None
        ts_gyro = None

        for frame in frames:
            if frame.get_profile().stream_type() == rs.stream.accel:
                accel_data = frame.as_motion_frame().get_motion_data()
                # coordinate transform so +Z is pointing up and +X is pointing out of the camera
                # to match helicopter dynamics simulation
                accel_data = np.array([accel_data.z, -accel_data.x, -accel_data.y]).copy()
                ts_accel = frame.get_timestamp() / 1000.
            elif frame.get_profile().stream_type() == rs.stream.gyro:
                gyro_data = frame.as_motion_frame().get_motion_data()
                gyro_data = np.array([gyro_data.z, -gyro_data.x, -gyro_data.y]).copy()
                ts_gyro = frame.get_timestamp() / 1000.

        self.time_queue.append(ts_gyro)

        if len(self.time_queue) < 2:
            self.last_accel = accel_data
            self.last_gyro = gyro_data
            return None
        else:
            accel_data = self.ema_factor * accel_data + (1 - self.ema_factor) * self.last_accel
            gyro_data = self.ema_factor * gyro_data + (1 - self.ema_factor) * self.last_gyro
            return accel_data, ts_accel, gyro_data, ts_gyro

    def stop(self):
        print("Closing D435i pipelines")
        if self.enable_motion:
            try:
                self.imu_pipeline.stop()
            except RuntimeError:
                pass

        try:
            self.pipeline.stop()
        except RuntimeError:
            pass
