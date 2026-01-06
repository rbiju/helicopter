from abc import ABC, abstractmethod

from helicopter.aircraft import Aircraft


class FlightController(ABC):
    def __init__(self, craft: Aircraft):
        self.craft = craft
        self.setpoint = 0.

    def update_setpoint(self, value):
        self.setpoint = value

    @abstractmethod
    def control(self, dt) -> float:
        # Output signal should always be on [-1, 1].
        # It is the job of the aircraft to scale it correctly
        raise NotImplementedError
