from dataclasses import dataclass


@dataclass
class SymaCommand:
    thrust: float
    pitch: float
    yaw: float
    trim: float = 0
    channel: int = 128

    def __post_init__(self):
        if not self.valid_range(self.thrust):
            raise ValueError(f"Invalid throttle value: {self.thrust}")
        if not self.valid_range(self.pitch):
            raise ValueError(f"Invalid pitch value: {self.pitch}")
        if not self.valid_range(self.yaw):
            raise ValueError(f"Invalid yaw value: {self.yaw}")
        if not self.valid_range(self.trim):
            raise ValueError(f"Invalid trim value: {self.trim}")

    @staticmethod
    def valid_range(value: float):
        if value < 0 or value > 1:
            return False
        return True

    @staticmethod
    def convert_to_int(value: float, max_value: int = 127, zero_value: int = 63) -> int:
        return max(int(value * (max_value - zero_value)) + zero_value, 0)

    def format(self) -> list[int]:
        return [self.channel,
                self.convert_to_int(self.yaw),
                self.convert_to_int(self.pitch),
                self.convert_to_int(self.thrust, zero_value=0),
                self.convert_to_int(self.trim)]

class SymaCommandFactory:
    @staticmethod
    def command(thrust: float,
                pitch: float,
                yaw: float,
                trim: float = 0,
                channel: int = 128):
        return SymaCommand(thrust, pitch, yaw, trim, channel)
