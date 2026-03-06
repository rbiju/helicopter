import serial
import struct

from .command import Command

READY_ACK = 129

class SymaRemoteControl:
    def __init__(self, port: str = '/dev/ttyUSB0', baudrate: int = 115200):
        self.arduino = serial.Serial(port=port, baudrate=baudrate, timeout=.1)

    def write(self, command: str):
        self.arduino.write(bytes(command, 'utf-8'))

    def read(self):
        data = self.arduino.readline()
        return data

    def send_command(self, command: Command):
        read_data = self.arduino.read(1)
        if read_data:
            read_data = struct.unpack('B', read_data)[0]
        if read_data == READY_ACK:
            cmd = command.format()
            for c in cmd:
                self.arduino.write(c.to_bytes(length=1, byteorder='big'))
