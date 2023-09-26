import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns


from manifolds.model_spaces import Hypersphere
from analyzers import (CoarseExtrinsicCurvatureAnalyzer,
                       DisplayCurvatureConvergence, DisplayCurvatureDistribution)
from util.util import gaussian

sphere = Hypersphere()
root = np.array([1, 0, 0])

# experiment parameters
# L = 10
# noises = np.full(L, 0.01) # 0.1 * np.arange(1, L)
# scales = np.full(L, 0.2) # 0.2 / np.arange(1, L)
# intensities = 5000 * np.arange(1, L+1) # ** 2

L = 1
noises = np.full(L, 0.2) # 0.1 * np.arange(1, L)
scales = np.full(L, 0.2) # 0.2 / np.arange(1, L)
intensities = 30000 * np.full(1, L+1) # ** 2

analyzer = CoarseExtrinsicCurvatureAnalyzer(sphere, root)
analyzer.analyze(scales, intensities, noises, num_runs=100)

# DisplayCurvatureConvergence.plot(analyzer, vary='intensity')
DisplayCurvatureDistribution.plot(analyzer, "curvature_distribution.png")
