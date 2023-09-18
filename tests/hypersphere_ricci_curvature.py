import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from manifolds.model_spaces import Hypersphere
from analyzers import CoarseRicciCurvatureAnalyzer, DisplayMidpointDistances, DisplayCurvatureDistribution


sphere = Hypersphere()
connectivities = 0.06 / np.arange(1, 2)
scales = 0.3 / np.arange(1, 2)
intensities = 10000 * np.arange(1, 2)
root = np.array([1, 0, 0])

# analyzer = CoarseRicciCurvatureAnalyzer(sphere, root)
# analyzer.analyze(connectivities, scales, intensities, algorithm="dijkstra", num_runs=1)

# DisplayMidpointDistances.plot(analyzer)
analyzer = CoarseRicciCurvatureAnalyzer(sphere, root)
analyzer.analyze(connectivities, scales, intensities, num_runs=200, method="optimization")

DisplayCurvatureDistribution.plot(analyzer, "ricci_curvature_distribution.png")
