from helicopter.utils import Profiler, SymaRemoteControl
from helicopter.vision import ErrorStateSquareRootUnscentedKalmanFilter, D435i
from helicopter.visualize import Visualizer


class FlightConductor:
    def __init__(self, aircraft: "Aircraft",
                 tracker: "Tracker",
                 controller: "Controller",
                 flight_plan: "FlightPlan",
                 board_handler: "BoardHandler",
                 remote: "SymaRemoteControl",
                 ukf: ErrorStateSquareRootUnscentedKalmanFilter,
                 camera: D435i,
                 visualizer: Visualizer,
                 profiler: Profiler) -> None:
        """

        Args:
            aircraft: State handler, owns the integrated aircraft state, and the physical model?
            tracker: Owns the visual tracking logic
            controller: The Brain
            flight_plan: Contains timestamped waypoints
            board_handler: Owns the board layout, finding aruco markers and populating the visualizer
            remote: Owns communication with the aircraft
            ukf: Fuses control inputs and vision for state estimation
            profiler:
        """
        self.aircraft = aircraft
        self.tracker = tracker
        self.controller = controller
        self.flight_plan = flight_plan
        self.board_handler = board_handler
        self.remote = remote
        self.ukf = ukf

        self.camera = camera

        self.profiler = profiler

        self.visualizer = visualizer


    def homing_sequence(self):
        """
        1. Scan for the aruco markers, populate the visualizer
        2. Initialize the camera position with the triangle point matcher, use averaged points.
            Might need to randomly jitter the rotor to make the markers visible.
        """
        pass

    def loop(self):
        """
        While flight path is incomplete:

        1. Compute error between waypoint and current state, frequently check if next waypoint can be ticked
        2. Calculate control input
        3. Apply input, simulate effect and feed to Kalman filter
        4. Use predicted location as starting point for ICP, feed visual update to KF
        5. Tick to next waypoint based on the time

        """
        pass
