from pathlib import Path

import numpy as np

from .point_matcher import TrianglePointMatcher
from .icp import ICP


class Tracker:
    def __init__(self, reference_points_path: str = 'assets/points_clouds/green_syma.npy'):
        reference_points_path = Path(__file__).parents[3] / reference_points_path
        self.reference_points = np.load(str(reference_points_path))

        self.matcher = TrianglePointMatcher(self.reference_points, n=5)
        self.icp = ICP(self.reference_points)
