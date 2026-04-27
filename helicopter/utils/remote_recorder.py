import serial

READY_ACK = 129


class SymaRemoteRecorder:
    def __init__(self, port: str = '/dev/ttyUSB0', baudrate: int = 115200):
        self.arduino = serial.Serial(port=port, baudrate=baudrate, timeout=.2)

    def read_command(self) -> list[int]:
        read_data = self.arduino.read(1)

        if read_data and read_data[0] == READY_ACK:
            commands = self.arduino.read(5)
            if len(commands) == 5:
                return list(commands)

        return []


class RemoteState:
    def __init__(self, recorder: SymaRemoteRecorder):
        self.recorder = recorder

        self.throttle = 0
        self.yaw = 63
        self.pitch = 63
        self.trim = 0

    def update(self, commands: list[int]):
        self.throttle = commands[3]
        self.pitch = commands[2]
        self.yaw = commands[1]
        self.trim = commands[4]


    def convert_to_float(self):
        thrust = self.throttle / 127.
        pitch = ((128 - (self.pitch + 1)) - 64.) / 128. * 2
        yaw = ((128 - (self.yaw + 1)) - 64.) / 128. * 2
        return [thrust, pitch, yaw]
