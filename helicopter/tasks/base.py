from abc import abstractmethod, ABC
from pathlib import Path
import re

from helicopter.configuration import HydraConfigurable


@HydraConfigurable
class Task(ABC):
    def __init__(self, **kwargs):
        pass

    @abstractmethod
    def run(self, **kwargs):
        raise NotImplementedError

    def debug_run(self, **kwargs):
        self.run(**kwargs)


def get_config_path(task):
    if isinstance(task, str):
        cls = task
    elif isinstance(task, Task):
        cls = task.__name__
    else:
        raise TypeError

    s1 = re.sub(r'(?<=[a-z0-9])([A-Z])', r'_\1', cls)
    file_name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', s1).lower()

    current_file = Path(__file__).resolve()
    root_dir = current_file.parents[2]
    yaml_path = root_dir / "configs" / "tasks" / f"{file_name}.yaml"
    return yaml_path
