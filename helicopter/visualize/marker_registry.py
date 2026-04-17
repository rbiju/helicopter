from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import trimesh

MUTUALLY_EXCLUSIVE_IDS = {"GameTable": [0, 1, 2]}


class MarkerModel(ABC):
    def __init__(self):
        self.objects_dir = Path(__file__).resolve().parents[2] / 'assets' / 'objects'
        self.obj_file = None

    @property
    @abstractmethod
    def id(self) -> int:
        raise NotImplementedError

    @property
    def obj_path(self) -> int:
        return self.objects_dir / self.obj_file

    @abstractmethod
    def mesh(self) -> trimesh.Trimesh:
        raise NotImplementedError

    @property
    @abstractmethod
    def marker_offset(self) -> np.ndarray:
        """
        Returns: np array representing marker coordinates with model center at origin

        """
        raise NotImplementedError

    @property
    def marker_rotation(self) -> Rotation:
        """

        Returns: Rotation representing marker rotation applied to default (normal vector facing -x)

        """
        return Rotation.from_quat(np.array([0, 0, 0, 1.0]))


class ModelRegistry:
    def __init__(self):
        self._classes = {}

    def register(self):
        def decorator(cls):
            if not issubclass(cls, MarkerModel):
                raise ValueError("Only MarkerModel objects should be registered here.")

            dummy_instance = cls()
            key = dummy_instance.id

            if key in self._classes:
                raise ValueError(f"Model with id '{key}' already registered.")

            self._classes[key] = cls
            return cls

        return decorator

    def get_class(self, key):
        return self._classes.get(key)

    def list_registered_classes(self):
        return list(self._classes.keys())


model_registry = ModelRegistry()


class GameTableModel(MarkerModel, ABC):
    def __init__(self):
        """
        Game table marker is mounted
        """
        super().__init__()
        self.marker_size_offset = 0.035
        self.obj_file = 'table/table.obj'

    def mesh(self) -> trimesh.Trimesh:
        mesh = trimesh.load_mesh(str(self.obj_path))
        mesh.apply_translation(np.array([0.0, 0.0, -0.05]))
        return mesh


@model_registry.register()
class GameTableModelTopSide(GameTableModel):
    def __init__(self):
        super().__init__()

    @property
    def id(self) -> int:
        return 0

    @property
    def marker_rotation(self) -> Rotation:
        return Rotation.from_euler('y', [90], degrees=True)

    @property
    def marker_offset(self) -> np.ndarray:
        return np.array([-0.355 + self.marker_size_offset,
                         -0.685 + self.marker_size_offset,
                         0.0])

@model_registry.register()
class GameTableModelShortSide(GameTableModel):
    def __init__(self):
        super().__init__()

    @property
    def id(self) -> int:
        return 1

    @property
    def marker_rotation(self) -> Rotation:
        return Rotation.from_euler('z', [90], degrees=True)

    @property
    def marker_offset(self) -> np.ndarray:
        return np.array([-0.355,
                         -0.685 - self.marker_size_offset,
                         - self.marker_size_offset])


@model_registry.register()
class GameTableModelLongSide(GameTableModel):
    def __init__(self):
        super().__init__()

    @property
    def id(self) -> int:
        return 2

    @property
    def marker_offset(self) -> np.ndarray:
        return np.array([-0.355 - self.marker_size_offset,
                         -0.685,
                         - self.marker_size_offset])
