from abc import ABC, abstractmethod


class FlightPlan(ABC):
    """
    This class is basically a generator of waypoints, where the tick function allows for serving the next waypoint
    """
    def __init__(self, dx: float):
        self.dx = dx

    @abstractmethod
    def flight_function(self):
        pass

    def tick(self):
        pass
