from pathlib import Path
from pyduinocli import Arduino


class ArduinoLoader:
    def __init__(self, sketch_path: str, port: str = "/dev/ttyUSB0"):
        cli_path = str(Path('~/.local/bin/arduino-cli').expanduser())
        self.arduino_cli = Arduino(cli_path)
        self.sketch_path = Path(__file__).parents[2] / 'arduino' / sketch_path
        self.hex_path = self.sketch_path / f"{sketch_path}.hex"

        self.port = port
        self.fqbn = "arduino:avr:nano:cpu=atmega328"

    def load(self):
        print(f"Uploading {self.hex_path} to {self.port}...")

        self.arduino_cli.upload(
            port=self.port,
            fqbn=self.fqbn,
            input_file=str(self.hex_path)
        )
