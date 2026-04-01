from collections import deque
from dataclasses import dataclass
from typing import Optional
import warnings

import numpy as np
import pyrealsense2 as rs


@dataclass
class VideoFrameOutput:
    depth_image: np.ndarray
    ir_image: np.ndarray
    color_image: np.ndarray
    depth_ts: float
    ir_ts: float
    color_ts: float
    laser_state: bool


class D435i:
    def __init__(self, accel_rate: int = 250,
                 gyro_rate: int = 200,
                 video_rate: int = 60,
                 video_resolution: tuple[int, int] | list[int] = (480, 640),
                 enable_rgb: bool = False,
                 enable_motion: bool = False,
                 global_time: bool = False,
                 projector_power: float = 150.,
                 toggle_projector: bool = False,
                 autoexpose: bool = True,
                 exposure_time: int = 2400,
                 autoexpose_rgb: bool = False,
                 exposure_time_rgb: int = 2400,
                 depth_preset: int = 4,
                 ema_accel: float = 1.0,
                 ema_gyro: float = 1.0,):
        self.ACCEL_RATE = accel_rate
        self.GYRO_RATE = gyro_rate

        # IR is always enabled
        if isinstance(video_resolution, list):
            if not len(video_resolution) == 2:
                raise ValueError("Video resolution must be a tuple of two integers.")
            video_resolution = (int(video_resolution[0]), int(video_resolution[1]))
        self.IR_RESOLUTION = video_resolution
        self.IR_RATE = video_rate
        self.DEPTH_RESOLUTION = video_resolution
        self.DEPTH_RATE = video_rate
        self.COLOR_RESOLUTION = video_resolution
        self.COLOR_RATE = video_rate

        self.enable_rgb = enable_rgb
        self.enable_motion = enable_motion

        # Sensor setup
        serial = self.get_device_serial()
        ctx = rs.context()
        device = self.get_device_from_serial(ctx, serial)
        depth_sensor = device.first_depth_sensor()
        depth_sensor.set_option(rs.option.visual_preset, depth_preset)
        self.depth_scale = depth_sensor.get_depth_scale()
        self.set_ir_exposure(depth_sensor, autoexposure=autoexpose, exposure_time=exposure_time)
        self.set_projector_power(depth_sensor, projector_power)

        self.color_ir_extrinsics = None
        if enable_rgb:
            color_sensor = device.first_color_sensor()
            self.set_color_exposure(color_sensor,
                                    autoexposure=autoexpose_rgb,
                                    exposure_time=exposure_time_rgb)
            self.color_ir_extrinsics = self.get_color_to_ir_extrinsics()

        self.global_time = global_time

        if toggle_projector:
            print("Laser projector toggle enabled")
            depth_sensor.set_option(rs.option.emitter_always_on, 0)
            depth_sensor.set_option(rs.option.emitter_on_off, 1)
        else:
            print('Emitter always on enabled')
            depth_sensor.set_option(rs.option.emitter_always_on, 1)

        depth_sensor.set_option(rs.option.frames_queue_size, 16)

        self.enable_motion = enable_motion
        if enable_motion:
            self.imu_pipeline, self.imu_config = self.get_imu_pipeline(serial)

        self.pipeline, self.config, self.intrinsics = self.get_camera_pipeline(serial, enable_rgb)

        self.align = rs.align(rs.stream.infrared)

        self.ema_accel = ema_accel
        self.ema_gyro = ema_gyro
        self.last_accel = None
        self.last_gyro = None

        self.time_queue = deque(maxlen=10)

    @property
    def last_imu_time(self):
        return self.time_queue[-2]

    @staticmethod
    def get_device_from_serial(ctx, serial):
        for dev in ctx.devices:
            if dev.get_info(rs.camera_info.serial_number) == serial:
                return dev
        raise RuntimeError(f"Device {serial} not found")

    @staticmethod
    def get_device_serial():
        ctx = rs.context()
        if not ctx.devices:
            raise RuntimeError("No RealSense device detected.")
        return ctx.devices[0].get_info(rs.camera_info.serial_number)

    @staticmethod
    def set_projector_power(depth_sensor: rs.depth_sensor, power: float):
        if 0. < power <= 360:
            print(f"Setting projector power to {power}")
            depth_sensor.set_option(rs.option.laser_power, power)
            depth_sensor.set_option(rs.option.emitter_enabled, 1)
        elif power == 0:
            print(f"Disabling projector (power was set to 0)")
            depth_sensor.set_option(rs.option.emitter_enabled, 0)
        else:
            warnings.warn(f"Specified projector power {power} is invalid. Must be between 0 and 360 inclusive"
                          f"Projector power setting skipped.")

    @staticmethod
    def set_ir_exposure(depth_sensor: rs.depth_sensor, autoexposure: bool, exposure_time: int):
        if not autoexposure:
            depth_sensor.set_option(rs.option.enable_auto_exposure, 0)
            depth_sensor.set_option(rs.option.exposure, exposure_time)

            print(f"Auto-exposure disabled and exposure set to {exposure_time}")
        else:
            depth_sensor.set_option(rs.option.enable_auto_exposure, 1)

            print("Auto-exposure enabled")

    @staticmethod
    def set_color_exposure(color_sensor: rs.color_sensor, autoexposure: bool, exposure_time: int):
        if not autoexposure:
            color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            color_sensor.set_option(rs.option.exposure, exposure_time)

            print(f"RGB Auto-exposure disabled and exposure set to {exposure_time}")
        else:
            color_sensor.set_option(rs.option.enable_auto_exposure, 1)

            print("RGB Auto-exposure enabled")

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

    def get_color_to_ir_extrinsics(self, color_format=rs.format.rgb8, ir_format=rs.format.y8, ir_index=1):
        if not self.enable_rgb:
            raise RuntimeError("RGB stream is not enabled. Cannot calculate color-to-IR extrinsics.")

        serial = self.get_device_serial()
        ctx = rs.context()
        devices = [dev for dev in ctx.devices if dev.get_info(rs.camera_info.serial_number) == serial]
        if not devices:
            raise RuntimeError(f"Device {serial} not found")

        device = devices[0]
        color_profile = None
        ir_profile = None

        color_height, color_width = self.COLOR_RESOLUTION
        ir_height, ir_width = self.IR_RESOLUTION

        for sensor in device.query_sensors():
            for profile in sensor.get_stream_profiles():
                if (profile.stream_type() == rs.stream.color and
                        profile.format() == color_format and
                        profile.fps() == self.COLOR_RATE):

                    if profile.is_video_stream_profile():
                        v_profile = profile.as_video_stream_profile()
                        if v_profile.width() == color_width and v_profile.height() == color_height:
                            color_profile = v_profile

                if (profile.stream_type() == rs.stream.infrared and
                        profile.stream_index() == ir_index and
                        profile.format() == ir_format and
                        profile.fps() == self.IR_RATE):

                    if profile.is_video_stream_profile():
                        v_profile = profile.as_video_stream_profile()
                        if v_profile.width() == ir_width and v_profile.height() == ir_height:
                            ir_profile = v_profile

                if color_profile and ir_profile:
                    return color_profile.get_extrinsics_to(ir_profile)

        raise RuntimeError(f"Extrinsics not found for device {serial}")

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

        motion_sensor.set_option(rs.option.enable_motion_correction, 1)
        motion_sensor.set_option(rs.option.frames_queue_size, 12)
        motion_sensor.set_option(rs.option.gyro_sensitivity, 4)

        return pipeline, config

    def start(self):
        if self.enable_motion:
            imu_profile = self.imu_pipeline.start(self.imu_config)
            device = imu_profile.get_device()
            motion_sensor = device.first_motion_sensor()
            if self.global_time:
                motion_sensor.set_option(rs.option.global_time_enabled, 1)
            else:
                motion_sensor.set_option(rs.option.global_time_enabled, 0)

            print("--- Active Motion Streams ---")
            for stream in imu_profile.get_streams():
                print(f"{stream.stream_type()} | FPS: {stream.fps()} | Format: {stream.format()}")

        profile = self.pipeline.start(self.config)
        device = profile.get_device()
        depth_sensor = device.first_depth_sensor()

        if self.global_time:
            depth_sensor.set_option(rs.option.global_time_enabled, 1)
        else:
            depth_sensor.set_option(rs.option.global_time_enabled, 0)

        print("--- Active Video Streams ---")
        for stream in profile.get_streams():
            if stream.is_video_stream_profile():
                v = stream.as_video_stream_profile()
                print(f"{v.stream_type()} | {v.width()}x{v.height()} | FPS: {v.fps()} | Format: {v.format()}")
            else:
                print(f"{stream.stream_type()} | FPS: {stream.fps()} | Format: {stream.format()}")

    def process_frames(self, frames: rs.composite_frame):
        aligned = self.align.process(frames)

        depth_frame = aligned.get_depth_frame()
        laser_state = (depth_frame.get_frame_metadata(rs.frame_metadata_value.frame_laser_power_mode) == 1)
        ir_frame = aligned.get_infrared_frame(1)
        color_frame = aligned.get_color_frame()

        if depth_frame:
            depth_image = np.asanyarray(depth_frame.get_data()).copy()
            depth_image = depth_image * self.depth_scale
            ts_depth = float(depth_frame.get_timestamp()) / 1000.
        else:
            depth_image = None
            ts_depth = None
        if ir_frame:
            ir_image = np.asanyarray(ir_frame.get_data()).copy()
            ts_ir = float(ir_frame.get_timestamp()) / 1000.
        else:
            ir_image = None
            ts_ir = None
        if color_frame:
            color_image = np.asanyarray(color_frame.get_data()).copy()
            ts_color = float(color_frame.get_timestamp()) / 1000.
        else:
            color_image = None
            ts_color = None

        return VideoFrameOutput(depth_image=depth_image,
                                depth_ts=ts_depth,
                                ir_image=ir_image,
                                ir_ts=ts_ir,
                                color_image=color_image,
                                color_ts=ts_color,
                                laser_state=laser_state)

    def process_imu_frames(self, frames, ema_accel: Optional[float] = None, ema_gyro: Optional[float] = None):
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
                ts_accel = float(frame.get_timestamp()) / 1000.
            elif frame.get_profile().stream_type() == rs.stream.gyro:
                gyro_data = frame.as_motion_frame().get_motion_data()
                gyro_data = np.array([gyro_data.z, -gyro_data.x, -gyro_data.y]).copy()
                ts_gyro = float(frame.get_timestamp()) / 1000.

        self.time_queue.append(ts_gyro)

        if len(self.time_queue) < 2:
            self.last_accel = accel_data
            self.last_gyro = gyro_data
            return None
        else:
            if ema_accel is None:
                ema_accel = self.ema_accel
            if ema_gyro is None:
                ema_gyro = self.ema_gyro

            accel_data = ema_accel * accel_data + (1 - ema_accel) * self.last_accel
            gyro_data = ema_gyro * gyro_data + (1 - ema_gyro) * self.last_gyro

            self.last_accel = accel_data
            self.last_gyro = gyro_data
            return accel_data, ts_accel, gyro_data, ts_gyro

    def stop(self):
        if self.enable_motion:
            try:
                self.imu_pipeline.stop()
            except RuntimeError:
                pass

        try:
            self.pipeline.stop()
        except RuntimeError:
            pass
        print("Camera pipelines stopped")
