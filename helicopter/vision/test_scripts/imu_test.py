import pyrealsense2 as rs
import numpy as np
from collections import deque


def initialize_camera():
    # start the frames pipe
    p = rs.pipeline()
    conf = rs.config()

    conf.disable_all_streams()
    conf.enable_stream(rs.stream.accel, rs.format.motion_xyz32f, 250)
    conf.enable_stream(rs.stream.gyro, rs.format.motion_xyz32f, 200)
    profile = p.start(conf)

    device = profile.get_device()
    motion_sensor = device.first_motion_sensor()
    motion_sensor.set_option(rs.option.global_time_enabled, 0)

    _ = profile
    return p


def gyro_data(gyro):
    return np.asarray([gyro.x, gyro.y, gyro.z])


def accel_data(accel):
    return np.asarray([accel.x, accel.y, accel.z])


_p = initialize_camera()
accel_queue = deque(maxlen=100)
gyro_queue = deque(maxlen=100)

try:
    while True:
        f = _p.wait_for_frames()

        # Process each frame independently
        for frame in f:
            if frame.is_motion_frame():
                motion_frame = frame.as_motion_frame()

                if frame.get_profile().stream_type() == rs.stream.accel:
                    _accel = accel_data(motion_frame.get_motion_data())
                    accel_timestamp = motion_frame.get_timestamp()
                    accel_queue.append((_accel, accel_timestamp))
                    print("accelerometer: ", _accel, " | timestamp: ", accel_timestamp, " | queue size: ",
                          len(accel_queue))

                elif frame.get_profile().stream_type() == rs.stream.gyro:
                    _gyro = gyro_data(motion_frame.get_motion_data())
                    gyro_timestamp = motion_frame.get_timestamp()
                    gyro_queue.append((_gyro, gyro_timestamp))
                    # print("gyro: ", _gyro, " | timestamp: ", gyro_timestamp, " | queue size: ", len(gyro_queue))

        if len(accel_queue) > 25:
            print('whoa!')

finally:
    _p.stop()
