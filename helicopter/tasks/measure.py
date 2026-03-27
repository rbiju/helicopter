import functools

from helicopter.configuration import HydraConfigurable
from helicopter.vision import UKFFactory

from .base import Task


@HydraConfigurable
class Measure(Task):
    def __init__(self,
                 ukf_factory: UKFFactory,
                 scanner: functools.partial):
        super().__init__()
        self.scanner = scanner(ukf=ukf_factory.filter())


    def run(self, configuration_path: str):
        self.scanner.scan()
