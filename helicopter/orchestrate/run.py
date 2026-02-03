"""

reference_pointcloud

# time average points here
points = point_getter.extract(ir_frame, depth_frame)
quat, t = triangle_point_getter.get_alignment(points)

while True:
    points = point_getter.extract(ir_frame, depth_frame)
    transformed_ref = kabsch.apply(quat, t, reference_pointcloud)
    dq, dt = icp.iterate(transformed_ref, points)

    quat = dq * quat
    t = t + dt


# multiprocessing setup

import multiprocessing as mp
import time
import numpy as np

def vision_process(shared_coords, new_data_evt):
    Runs at ~30 FPS (or whatever the camera allows).
    Extracts (x, y, z) and notifies the controller.

    while True:
        # 1. Capture and process image (CV Pipeline)
        raw_x, raw_y, raw_z = detect_helicopter()

        # 2. Write to shared memory with a lock
        with shared_coords.get_lock():
            shared_coords[0] = raw_x
            shared_coords[1] = raw_y
            shared_coords[2] = raw_z

        # 3. Signal to the controller that fresh data is waiting
        new_data_evt.set()

def control_process(shared_coords, new_data_evt):
    Runs at 100Hz.
    Owns the Kalman Filter and PID.
    # Initialize KF and PID locally (no GIL sharing issues)
    kf = KalmanFilter()
    pid = PIDController()
    last_time = time.time()

    while True:
        now = time.time()
        dt = now - last_time

        # --- KALMAN STEP 1: PREDICTION ---
        # Predict where we are based on physics + the LAST command sent
        kf.predict(dt, last_command)

        # --- KALMAN STEP 2: CORRECTION (Asynchronous) ---
        # Check if the CV process has posted a new model_training
        if new_data_evt.is_set():
            with shared_coords.get_lock():
                model_training = list(shared_coords)

            new_data_evt.clear() # Reset the flag
            kf.update(model_training) # Correct the filter with real data

        # --- CONTROL STEP ---
        filtered_state = kf.get_state()
        command = pid.compute(filtered_state)
        send_to_arduino(command)

        last_command = command
        last_time = now

        # Precise 100Hz timing
        time.sleep(max(0, 0.01 - (time.time() - now)))

if __name__ == "__main__":
    # 'd' stands for double precision float
    shared_coords = mp.Array('d', [0.0, 0.0, 0.0])
    new_data_evt = mp.Event()

    p_vision = mp.Process(target=vision_process, args=(shared_coords, new_data_evt))
    p_control = mp.Process(target=control_process, args=(shared_coords, new_data_evt))

    p_vision.start()
    p_control.start()

"""