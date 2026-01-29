import csv
from pathlib import Path

import numpy as np
import quaternion


class StateLogger:
    def __init__(self, save_dir="../../../notebooks"):
        self.save_dir = Path(save_dir)
        self.data = []
        self.headers = [
            'timestamp', 'event',
            'qw', 'qx', 'qy', 'qz',
            'px', 'py', 'pz',
            'vx', 'vy', 'vz',
            'b_ax', 'b_ay', 'b_az',
            'b_gx', 'b_gy', 'b_gz'
        ]

        self.imu_data = []
        self.imu_headers = ['timestamp',
                            'ax', 'ay', 'az',
                            'gx', 'gy', 'gz',]

        self.vision_data = []
        self.vision_headers = ['timestamp',
                               'qw', 'qx', 'qy', 'qz',
                               'px', 'py', 'pz']

    def log_state(self, timestamp: float, event: str, state_vector: np.ndarray):
        s = state_vector.flatten()

        row = [
            f"{timestamp:.6f}",
            event,
            f"{s[0]:.6f}", f"{s[1]:.6f}", f"{s[2]:.6f}", f"{s[3]:.6f}",
            f"{s[4]:.6f}", f"{s[5]:.6f}", f"{s[6]:.6f}",
            f"{s[7]:.6f}", f"{s[8]:.6f}", f"{s[9]:.6f}",
            f"{s[10]:.6f}", f"{s[11]:.6f}", f"{s[12]:.6f}",
            f"{s[13]:.6f}", f"{s[14]:.6f}", f"{s[15]:.6f}"
        ]
        self.data.append(row)

    def log_imu(self, timestamp: float, accel: np.ndarray, gyro: np.ndarray):
        row = [
            f"{timestamp:.6f}",
            f"{accel[0]:.6f}", f"{accel[1]:.6f}", f"{accel[2]:.6f}",
            f"{gyro[0]:.6f}", f"{gyro[1]:.6f}", f"{gyro[2]:.6f}",
        ]
        self.imu_data.append(row)

    def log_vision(self, timestamp: float, quat: quaternion.quaternion, translation: np.ndarray):
        row = [
            f"{timestamp:.6f}",
            f"{quat.w:.6f}", f"{quat.x:.6f}", f"{quat.y:.6f}", f"{quat.z:.6f}",
            f"{translation[0]:.6f}", f"{translation[1]:.6f}", f"{translation[2]:.6f}",
        ]
        self.vision_data.append(row)

    def save(self):
        print(f"Saving logs to {self.save_dir} with {len(self.data)} entries...")

        log_path = self.save_dir / "log.csv"
        imu_path = self.save_dir / "imu_log.csv"
        vision_path = self.save_dir / "vision_log.csv"

        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
            writer.writerows(self.data)

        with open(imu_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.imu_headers)
            writer.writerows(self.imu_data)

        with open(vision_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.vision_headers)
            writer.writerows(self.vision_data)

        print("Logs saved.")
