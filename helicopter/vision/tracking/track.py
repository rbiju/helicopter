import numpy as np
import quaternionic

from .point_matcher import TrianglePointMatcher
from .icp import ICP


class Tracker:
    def __init__(self, reference_points: np.ndarray):
        self.reference_points = reference_points
        self.matcher = TrianglePointMatcher(reference_points, n=5)
        self.icp = ICP(reference_points)

        self.q = quaternionic.array([1.0, 0.0, 0.0, 0.0])
        self.t = np.array([0.0, 0.0, 0.0])

    def get_initial_orientation(self, sample_points: np.ndarray):
        # Since this will be done when the helicopter is stationary, sample points should be a time average to reduce noise
        q, t = self.matcher.get_alignment(sample_points)

        return quaternionic.array(q), t

    def orientation(self, q_old, t_old, sample_points: np.ndarray):
        q, t = self.icp.iterate(q_old, t_old, sample_points)

        return quaternionic.array(q), t
