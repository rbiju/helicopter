"""
Module for computing point cloud orientation using triangle matching and ICP.
"""
from __future__ import annotations
import numpy
import numpy.typing
import typing

__all__: list[str] = ['ICP', 'TrianglePointMatcher']
class ICP:
    def __init__(self, max_iter: typing.SupportsInt, distance_threshold: typing.SupportsFloat, etol: typing.SupportsFloat) -> None:
        """
        Create an ICP solver.
        
        Args:
            max_iter: Maximum number of iterations
            distance_threshold: Distance beyond which correspondence will not be calculated.
            etol: Alignment error at which to stop iterating early
        """
    def get_correspondences(self, ref_points: typing.Annotated[numpy.typing.ArrayLike, numpy.float32, "[m, 3]"], sample_points: typing.Annotated[numpy.typing.ArrayLike, numpy.float32, "[m, 3]"]) -> tuple[list[int], list[int]]:
        ...
    def iterate(self, ref_points: typing.Annotated[numpy.typing.ArrayLike, numpy.float32, "[m, 3]"], sample_points: typing.Annotated[numpy.typing.ArrayLike, numpy.float32, "[m, 3]"]) -> tuple[..., ..., typing.Annotated[numpy.typing.NDArray[numpy.float32], "[3, 1]"]]:
        """
        Performs ICP.
        
        Args:
            ref_points: Points to be transformed (Nx3 Eigen/Numpy)
            sample_points: Measured points (Nx3 Eigen/Numpy)
        
        Returns:
            tuple[numpy.ndarray, numpy.ndarray] : (quaternion_xyzw, translation_xyz)
        """
class TrianglePointMatcher:
    def __init__(self, reference_points: typing.Annotated[numpy.typing.ArrayLike, numpy.float32, "[m, 3]"], n: typing.SupportsFloat = 5) -> None:
        """
        Create a TrianglePointMatcher with reference points.
        
        Args:
            reference_points: Nx3 numpy array of reference point coordinates
            n: Number of triangle candidates to try when matching (lower is faster)
        """
    def get_alignment(self, sample_points: typing.Annotated[numpy.typing.ArrayLike, numpy.float32, "[m, 3]"]) -> tuple[..., ..., typing.Annotated[numpy.typing.NDArray[numpy.float32], "[3, 1]"]]:
        """
        Find rotation + translation that best explains a rigid transform from reference to sample.
        
        Args:
            sample_points: Nx3 numpy array of measured point coordinates
        
        Returns:
            tuple[numpy.ndarray, numpy.ndarray] : (quaternion_xyzw, translation_xyz)
        """
