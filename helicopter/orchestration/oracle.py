from .flightplan import FlightPlan


class Oracle:
    """
    Glues together different flight plans, tracks flying state, and communicates with the orchestrator on flight status
    """
    def __init__(self):
        pass

    def add_flightplan(self, flightplan: FlightPlan):
        pass

    def get_next_flightplan(self):
        pass
