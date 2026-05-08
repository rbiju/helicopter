import csv
from collections import deque
from pathlib import Path
import numpy as np


class TrackerStateLogger:
    def __init__(self, save_dir='tracking'):
        self.save_dir = self.save_dir = Path(__file__).parents[3] / 'notebooks/logs' / save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.data = deque()
        self.headers = [
            'timestamp', 'event',
            'qx', 'qy', 'qz', 'qw',
            'px', 'py', 'pz',
            'ox', 'oy', 'oz',
            'vx', 'vy', 'vz',
            'battery', 'trim',
            'actual_throttle', 'actual_pitch', 'actual_yaw',
            'sys_state'
        ]

        self.cmd_data = deque()
        self.cmd_headers = ['timestamp', 'thrust', 'pitch', 'yaw']

    def log_state(self, event: str, state_vector: np.ndarray):
        s = state_vector.flatten()

        row = [
            f"{s[19]:.8f}",
            event,
            f"{s[0]:.8f}", f"{s[1]:.8f}", f"{s[2]:.8f}", f"{s[3]:.8f}",
            f"{s[4]:.8f}", f"{s[5]:.8f}", f"{s[6]:.8f}",
            f"{s[7]:.8f}", f"{s[8]:.8f}", f"{s[9]:.8f}",
            f"{s[10]:.8f}", f"{s[11]:.8f}", f"{s[12]:.8f}",
            f"{s[13]:.8f}", f"{s[14]:.8f}",
            f"{s[15]:.8f}", f"{s[16]:.8f}", f"{s[17]:.8f}",
            f"{s[18]:.8f}"
        ]

        self.data.append(row)

    def log_commands(self, timestamp: float, commands: np.ndarray):
        row = [
            f"{timestamp:.8f}",
            f"{commands[0]:.8f}", f"{commands[1]:.8f}", f"{commands[2]:.8f}"
        ]
        self.cmd_data.append(row)

    @staticmethod
    def write_file(save_path: Path, headers, data):
        with open(save_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(data)

    def save(self):
        print(f"Saving logs to {self.save_dir}...")

        log_path = self.save_dir / "log.csv"
        cmd_path = self.save_dir / "commands_log.csv"

        self.write_file(log_path, self.headers, list(self.data))
        self.write_file(cmd_path, self.cmd_headers, list(self.cmd_data))

        print(f"Saved {len(self.data)} state entries and {len(self.cmd_data)} command entries.")