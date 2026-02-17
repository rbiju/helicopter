import csv
from pathlib import Path

import numpy as np
import quaternion


class StateLogger:
    def __init__(self, save_dir="../../../notebooks"):
        self.save_dir = Path(save_dir)
        self.warmup_data = []
        self.data = []
        self.headers = [
            'timestamp', 'event',
            'qx', 'qy', 'qz', 'qw',
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

    def log_state(self, timestamp: float, event: str, state_vector: np.ndarray, warmup: bool = False):
        s = state_vector.flatten()

        row = [
            f"{timestamp:.8f}",
            event,
            f"{s[0]:.8f}", f"{s[1]:.8f}", f"{s[2]:.8f}", f"{s[3]:.8f}",
            f"{s[4]:.8f}", f"{s[5]:.8f}", f"{s[6]:.8f}",
            f"{s[7]:.8f}", f"{s[8]:.8f}", f"{s[9]:.8f}",
            f"{s[10]:.8f}", f"{s[11]:.8f}", f"{s[12]:.8f}",
            f"{s[13]:.8f}", f"{s[14]:.8f}", f"{s[15]:.8f}"
        ]
        if warmup:
            self.warmup_data.append(row)
        else:
            self.data.append(row)

    def log_imu(self, timestamp: float, accel: np.ndarray, gyro: np.ndarray):
        row = [
            f"{timestamp:.8f}",
            f"{accel[0]:.8f}", f"{accel[1]:.8f}", f"{accel[2]:.8f}",
            f"{gyro[0]:.8f}", f"{gyro[1]:.8f}", f"{gyro[2]:.8f}",
        ]
        self.imu_data.append(row)

    def log_vision(self, timestamp: float, quat: quaternion.quaternion, translation: np.ndarray):
        row = [
            f"{timestamp:.8f}",
            f"{quat.w:.8f}", f"{quat.x:.8f}", f"{quat.y:.8f}", f"{quat.z:.8f}",
            f"{translation[0]:.8f}", f"{translation[1]:.8f}", f"{translation[2]:.8f}",
        ]
        self.vision_data.append(row)

    @staticmethod
    def write_file(save_path: Path, headers, data):
        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(data)

    def save(self):
        print(f"Saving logs to {self.save_dir} with {len(self.data)} entries...")

        warmup_path = self.save_dir / "warmup.csv"
        log_path = self.save_dir / "log.csv"
        imu_path = self.save_dir / "imu_log.csv"
        vision_path = self.save_dir / "vision_log.csv"

        self.write_file(warmup_path, self.headers, self.warmup_data)
        self.write_file(log_path, self.headers, self.data)
        self.write_file(imu_path, self.imu_headers, self.imu_data)
        self.write_file(vision_path, self.vision_headers, self.vision_data)

        print("Logs saved.")
