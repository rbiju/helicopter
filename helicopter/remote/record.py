import serial
import time

from .base import RemoteState, RecordingPacket, RemoteThread, READY_ACK

class RecorderRemoteState(RemoteState):
    def __init__(self,):
        super().__init__()

    def update(self, commands: RecordingPacket):
        self.channel = commands.channel
        self.yaw = commands.yaw
        self.pitch = commands.pitch
        self.throttle = commands.throttle
        self.trim = commands.trim


class SymaRemoteRecorder:
    def __init__(self, port: str = '/dev/ttyUSB0', baudrate: int = 115200):
        self.arduino = serial.Serial(port=port, baudrate=baudrate, timeout=.2)
        self.remote_state = RecorderRemoteState()

    def read_command(self) -> list[int]:
        read_data = self.arduino.read(1)

        if read_data and read_data[0] == READY_ACK:
            commands = self.arduino.read(5)
            if len(commands) == 5:
                return list(commands)

        return []

    def shutdown(self):
        self.arduino.close()


class RemoteRecorderThread(RemoteThread):
    def __init__(self, remote_recorder: SymaRemoteRecorder = SymaRemoteRecorder()):
        super().__init__()
        self.remote_recorder = remote_recorder

    def run(self):
        while self.running:
            commands = self.remote_recorder.read_command()

            if len(commands) == 5:
                commands = RecordingPacket.from_list(commands)
                with self.lock:
                    self.remote_recorder.remote_state.update(commands)

            time.sleep(0.001)

    def get_commands(self):
        with self.lock:
            return self.remote_recorder.remote_state.convert_to_float()

    def shutdown(self):
        self.remote_recorder.shutdown()