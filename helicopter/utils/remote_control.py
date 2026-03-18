import serial
import struct

READY_ACK = 129

class SymaRemoteControl:
    def __init__(self, port: str = '/dev/ttyUSB0', baudrate: int = 115200):
        self.arduino = serial.Serial(port=port, baudrate=baudrate, timeout=.1)

    def write(self, command: str):
        self.arduino.write(bytes(command, 'utf-8'))

    def read(self):
        data = self.arduino.readline()
        return data

    def send_command(self, commands: list[int]) -> bool:
        read_data = self.arduino.read(1)
        if read_data:
            read_data = struct.unpack('B', read_data)[0]
        if read_data == READY_ACK:
            for command in commands:
                self.arduino.write(command.to_bytes(length=1, byteorder='big'))

            return True
        else:
            return False

    def shutdown(self):
        commands = [0, 63, 63, 0, 0]
        for command in commands:
            self.arduino.write(command.to_bytes(length=1, byteorder='big'))

        self.arduino.close()
