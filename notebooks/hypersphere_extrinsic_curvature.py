import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns


from manifolds.model_spaces import Hypersphere
from analyzers.analyzers import CoarseExtrinsicCurvatureAnalyzer


sphere = Hypersphere()
scales = 0.3 / np.arange(1, 2)
intensities = 10000 * np.arange(1, 2)
root = np.array([1, 0, 0])

analyzer = CoarseExtrinsicCurvatureAnalyzer(sphere, root)
analyzer.analyze(scales, intensities, noise=0.1, num_runs=1)
sns.histplot([float(c) for c in analyzer.sample_curvatures[0]])
plt.show()
