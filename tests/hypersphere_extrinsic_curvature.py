import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns


from manifolds.model_spaces import Hypersphere
from analyzers.analyzers import CoarseExtrinsicCurvatureAnalyzer, DisplayCurvatureConvergence
from util.util import gaussian

sphere = Hypersphere()
root = np.array([1, 0, 0])

# experiment parameters
# L = 10
# noises = np.full(L, 0.01) # 0.1 * np.arange(1, L)
# scales = np.full(L, 0.2) # 0.2 / np.arange(1, L)
# intensities = 5000 * np.arange(1, L+1) # ** 2

L = 1
noises = np.full(L, 0.01) # 0.1 * np.arange(1, L)
scales = np.full(L, 0.2) # 0.2 / np.arange(1, L)
intensities = 10000 * np.full(1, L+1) # ** 2

analyzer = CoarseExtrinsicCurvatureAnalyzer(sphere, root)
analyzer.analyze(scales, intensities, noises, num_runs=1000)

# DisplayCurvatureConvergence.plot(analyzer, vary='intensity')

# empirical distribution of extrinsic curvature
curvatures = np.array([float(c) for c in analyzer.sample_curvatures[-1]])
fig, ax = plt.subplots()
m = np.mean(curvatures)
s = np.std(curvatures)
sns.histplot(curvatures, stat='density')
x = np.linspace(-3 * s, 3 * s, 6000)
sns.lineplot(x=x, y=gaussian(x, m, s), linewidth=3)
plt.ylabel("Density")
plt.xlabel("Curvature")
plt.show()
