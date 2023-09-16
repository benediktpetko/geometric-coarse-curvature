import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns


from manifolds.model_spaces import Hypersphere
from analyzers.analyzers import CoarseExtrinsicCurvatureAnalyzer, DisplayCurvatureConvergence


sphere = Hypersphere()
L = 4
noises = np.full(L, 0.01) # 0.1 * np.arange(1, L)
scales = np.full(L, 0.2) # 0.2 / np.arange(1, L)
intensities = 1000 * 2 ** np.arange(1, L+1) # ** 2
root = np.array([1, 0, 0])

analyzer = CoarseExtrinsicCurvatureAnalyzer(sphere, root)
analyzer.analyze(scales, intensities, noises, num_runs=300)

DisplayCurvatureConvergence.plot(analyzer, vary='intensity')
# sns.histplot([float(c) for c in analyzer.sample_curvatures[0]])
# plt.show()
