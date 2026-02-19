import csv
from pathlib import Path
from collections import deque

import numpy as np


class StateLogger:
    def __init__(self, save_dir="../../../notebooks/logs"):
        self.save_dir = Path(save_dir)
        self.data = deque()
        self.headers = [
            'timestamp', 'event',
            'qx', 'qy', 'qz', 'qw',
            'px', 'py', 'pz',
            'vx', 'vy', 'vz',
            'b_ax', 'b_ay', 'b_az',
            'b_gx', 'b_gy', 'b_gz'
        ]

        self.imu_data = deque()
        self.imu_headers = ['timestamp',
                            'ax', 'ay', 'az',
                            'gx', 'gy', 'gz',]

    def log_state(self, timestamp: float, event: str, state_vector: np.ndarray):
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

        self.data.append(row)

    def log_imu(self, timestamp: float, accel: np.ndarray, gyro: np.ndarray):
        row = [
            f"{timestamp:.8f}",
            f"{accel[0]:.8f}", f"{accel[1]:.8f}", f"{accel[2]:.8f}",
            f"{gyro[0]:.8f}", f"{gyro[1]:.8f}", f"{gyro[2]:.8f}",
        ]
        self.imu_data.append(row)

    @staticmethod
    def write_file(save_path: Path, headers, data):
        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(data)

    def save(self):
        print(f"Saving logs to {self.save_dir} with {len(self.data)} entries...")

        log_path = self.save_dir / "log.csv"
        imu_path = self.save_dir / "imu_log.csv"

        self.write_file(log_path, self.headers, list(self.data))
        self.write_file(imu_path, self.imu_headers, list(self.imu_data))

        print("Logs saved.")
