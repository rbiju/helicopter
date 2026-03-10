class FlightState:
    def __init__(self, name: str):
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def engine_on(self) -> bool:
        return True

class FlightStateRegistry:
    def __init__(self):
        self._classes = {}

    def register(self):
        def decorator(cls):
            if not issubclass(cls, FlightState):
                raise ValueError("Only FlightState objects should be registered.")
            key = cls.name
            if key in self._classes:
                raise ValueError(f"FlightState with key '{key}' already registered.")
            self._classes[key] = cls
            return cls

        return decorator

    def get_class(self, key):
        return self._classes.get(key)

    def list_registered_classes(self):
        return list(self._classes.keys())


flight_state_registry = FlightStateRegistry()


@flight_state_registry.register()
class Init(FlightState):
    def __init__(self):
        super().__init__("Init")

    @property
    def engine_on(self):
        return False


@flight_state_registry.register()
class SpinUp(FlightState):
    def __init__(self):
        super().__init__("SpinUp")


@flight_state_registry.register()
class Takeoff(FlightState):
    def __init__(self):
        super().__init__("Takeoff")


@flight_state_registry.register()
class Waypoint(FlightState):
    def __init__(self):
        super().__init__("Waypoint")


@flight_state_registry.register()
class Hover(FlightState):
    def __init__(self):
        super().__init__("Hover")


@flight_state_registry.register()
class Landing(FlightState):
    def __init__(self):
        super().__init__("Landing")


@flight_state_registry.register()
class Idle(FlightState):
    def __init__(self):
        super().__init__("Idle")

    @property
    def engine_on(self):
        return False


@flight_state_registry.register()
class Done(FlightState):
    def __init__(self):
        super().__init__("Idle")

    @property
    def engine_on(self):
        return False
