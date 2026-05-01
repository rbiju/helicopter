import serial
import struct
import time

from .base import RemoteState, ControlPacket, RemoteThread, READY_ACK


class ControllerRemoteState(RemoteState):
    def __init__(self,):
        super().__init__()

    def update(self, commands: ControlPacket):
        self.throttle = int(commands.throttle * 127.)
        self.pitch = int(commands.pitch * 64. - 63)
        self.yaw = int(commands.yaw * 64. + 63)


class SymaRemoteControl:
    def __init__(self, port: str = '/dev/ttyUSB0', baudrate: int = 115200):
        self.arduino = serial.Serial(port=port, baudrate=baudrate, timeout=.1)
        self.remote_state = ControllerRemoteState()

        self.most_recently_sent = self.remote_state.convert_to_float()

    def send_command(self, commands: list[int]):
        read_data = self.arduino.read(1)
        if read_data:
            read_data = struct.unpack('B', read_data)[0]
        if read_data == READY_ACK:
            for command in commands:
                self.arduino.write(command.to_bytes(length=1, byteorder='big'))

            self.most_recently_sent = self.remote_state.convert_to_float()

    def shutdown(self):
        commands = [0, 63, 63, 0, 0]
        for command in commands:
            self.arduino.write(command.to_bytes(length=1, byteorder='big'))

        self.arduino.close()


class RemoteControlThread(RemoteThread):
    def __init__(self, rc: SymaRemoteControl):
        super().__init__()
        self.rc = rc

    def run(self):
        while self.running:
            with self.lock:
                commands = self.rc.remote_state.as_list()
            self.rc.send_command(commands)
            time.sleep(0.001)

    def update(self, command: ControlPacket):
        with self.lock:
            self.rc.remote_state.update(command)

    def most_recently_sent(self):
        with self.lock:
            return self.rc.remote_state.convert_to_float()

    def shutdown(self):
        self.rc.shutdown()
