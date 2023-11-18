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
# noises =  # np.full(L, 0.01) # 0.1 * np.arange(1, L)
# scales = np.full(L, 0.2) # 0.2 / np.arange(1, L)
# intensities = 5000 * np.arange(1, L+1) # ** 2

# L = 10
# noises = 0.2 / np.arange(1, L+1) # np.full(L, 0.01) # 0.12 * np.arange(1, L)
# noises = [0.25, 0.2, 0.15, 0.9, 0.075, 0.05, 0.025, 0.01, 0.005, 0]
# scales = np.full(L, 0.2) # 0.2 / np.arange(1, L)
# intensities = 40000 * np.full(L, 1) # ** 2

L = 1
noises = np.full(L, 0.2) # 0.1 * np.arange(1, L)
scales = np.full(L, 0.2) # 0.2 / np.arange(1, L)
intensities = 30000 * np.full(L, 1) # ** 2

analyzer = CoarseExtrinsicCurvatureAnalyzer(sphere, root)
analyzer.analyze(scales, intensities, noises, num_runs=300)

# DisplayCurvatureConvergence.plot(analyzer, filename='curvature_convergence_noise.png', vary='noise')
DisplayCurvatureDistribution.plot(analyzer, "curvature_distribution.png")
# sphere.plot(analyzer.point_cloud.noisy_points_subset[:100])
