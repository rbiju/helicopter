import serial
import time

from .base import RemoteState, ControlPacket, RemoteThread, READY_ACK


class ControllerRemoteState(RemoteState):
    def __init__(self,):
        super().__init__()

    @staticmethod
    def clip(value: int) -> int:
        return max(min(value, 127), 0)

    def update(self, commands: ControlPacket):
        self.throttle = self.clip(int(commands.throttle * 127.))
        self.pitch = self.clip(int(63 - (commands.pitch * 64.)))
        self.yaw = self.clip(int(commands.yaw * 64. + 63))


class SymaRemoteControl:
    def __init__(self, port: str = '/dev/ttyUSB0', baudrate: int = 1000000):
        self.arduino = serial.Serial(port=port, baudrate=baudrate, timeout=0)
        self.remote_state = ControllerRemoteState()

        self.most_recently_sent = self.remote_state.convert_to_float()

    def send_command(self, commands: list[int]):
        if self.arduino.in_waiting > 0:
            read_data = self.arduino.read(self.arduino.in_waiting)
            if READY_ACK in read_data:
                self.arduino.write(bytes(commands))

                self.most_recently_sent = self.remote_state.convert_to_float()

    def shutdown(self):
        commands = self.remote_state.default
        self.arduino.write(bytes(commands))

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

    def update(self, command: ControlPacket | None):
        if command is not None:
            with self.lock:
                self.rc.remote_state.update(command)

    def most_recently_sent(self):
        with self.lock:
            return self.rc.most_recently_sent

    def shutdown(self):
        self.rc.shutdown()
