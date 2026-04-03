from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

import trimesh


class Model(ABC):
    def __init__(self, obj_path: str):
        self.obj_path = Path(__file__).parent.parent.parent / obj_path

    @staticmethod
    def transformation_matrix(rotation: Rotation, translation: np.ndarray) -> np.ndarray:
        transform = np.zeros((4, 4))
        transform[:3, :3] = Rotation.as_matrix(rotation)
        transform[3, :3] = translation
        transform[3, 3] = 1.0

        return transform

    @abstractmethod
    def mesh(self) -> trimesh.Trimesh:
        raise NotImplementedError


class D435iModel(Model):
    def __init__(self, obj_path: str = 'assets/objects/camera/camera.obj'):
        super().__init__(obj_path)

    def mesh(self) -> trimesh.Trimesh:
        mesh = trimesh.load_mesh(self.obj_path)
        centroid = mesh.centroid
        mesh.apply_translation(-centroid)
        mesh.apply_scale(0.001)

        rotA = Rotation.from_rotvec(np.array([0.0, 0.0, np.pi / 2]))
        rotB = Rotation.from_rotvec(np.array([np.pi / 2, 0.0, 0.0]))
        rotation = rotA * rotB

        transform = self.transformation_matrix(rotation, np.array([0.0, 0.0, 0.0]))

        mesh.apply_transform(transform)

        return mesh


class HelicopterModel(Model):
    def __init__(self, obj_path: str = 'assets/objects/helicopter/helicopter.obj'):
        super().__init__(obj_path)

    def mesh(self) -> trimesh.Trimesh:
        mesh = trimesh.load_mesh(self.obj_path)
        centroid = mesh.centroid
        mesh.apply_translation(-centroid)
        mesh.apply_scale(3.37e-5)

        rotation = Rotation.from_euler('XZ', [-np.pi / 2, np.pi])

        transform = self.transformation_matrix(rotation, np.array([0.0, 0.0, 0.0]))

        mesh.apply_transform(transform)

        return mesh
