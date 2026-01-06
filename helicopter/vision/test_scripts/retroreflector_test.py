import pyrealsense2 as rs
import numpy as np

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 60)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)

pipeline.start(config)

try:
    for _ in range(500):
        pipeline.wait_for_frames()

    frames = pipeline.wait_for_frames()
    ir_frame = frames.get_infrared_frame(1)
    ir_array = np.asanyarray(ir_frame.get_data())
    print(f"\nIR range: {ir_array.min()} - {ir_array.max()} mm")

finally:
    pipeline.stop()
