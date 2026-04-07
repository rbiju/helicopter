from abc import ABC, abstractmethod

import numpy as np
from scipy.spatial.transform import Rotation

import trimesh

MUTUALLY_EXCLUSIVE_IDS = {"GameTable": [0, 1, 2]}


class MarkerModel(ABC):
    def __init__(self):
        pass

    @property
    @abstractmethod
    def id(self) -> int:
        raise NotImplementedError

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

        Returns: Rotation representing marker rotation from default (normal vector facing -x)

        """
        return Rotation.from_quat(np.array([0, 0, 0, 1.0]))

class ModelRegistry:
    def __init__(self):
        self._classes = {}

    def register(self):
        def decorator(cls):
            if not issubclass(cls, MarkerModel):
                raise ValueError("Only MarkerModel objects should be registered here.")
            key = cls.id
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
        self.marker_size_offset = 0.025

    def mesh(self) -> trimesh.Trimesh:
        obj_path: str = 'assets/objects/table/table.obj'

        mesh = trimesh.load_mesh(obj_path)
        return mesh


@model_registry.register()
class GameTableModelTopSide(GameTableModel):
    def __init__(self):
        super().__init__()

    @property
    def id(self) -> int:
        return 0

    def marker_rotation(self) -> Rotation:
        return Rotation.from_euler('Y', [90], degrees=True)

    def marker_offset(self) -> np.ndarray:
        return np.array([-0.355 + self.marker_size_offset,
                         -0.685 + self.marker_size_offset,
                         0.0 - self.marker_size_offset])

@model_registry.register()
class GameTableModelShortSide(GameTableModel):
    def __init__(self):
        super().__init__()

    @property
    def id(self) -> int:
        return 1

    def marker_rotation(self) -> Rotation:
        return Rotation.from_euler('Z', [90], degrees=True)

    def marker_offset(self) -> np.ndarray:
        return np.array([-0.355,
                         -0.685 - self.marker_size_offset,
                         0.025 - self.marker_size_offset])


@model_registry.register()
class GameTableModelLongSide(GameTableModel):
    def __init__(self):
        super().__init__()

    @property
    def id(self) -> int:
        return 2

    def marker_offset(self) -> np.ndarray:
        return np.array([-0.355 - self.marker_size_offset,
                         -0.685,
                         0.025 - self.marker_size_offset])
