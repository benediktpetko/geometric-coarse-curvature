import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from manifolds.model_spaces import Hypersphere
from analyzers.analyzers import CoarseRicciCurvatureAnalyzer, DisplayMidpointDistances


sphere = Hypersphere()
connectivities = 0.06 / np.arange(1, 2)
scales = 0.3 / np.arange(1, 2)
intensities = 10000 * np.arange(1, 2)
root = np.array([1, 0, 0])

# analyzer = CoarseRicciCurvatureAnalyzer(sphere, root)
# analyzer.analyze(connectivities, scales, intensities, algorithm="dijkstra", num_runs=1)

# DisplayMidpointDistances.plot(analyzer)
analyzer = CoarseRicciCurvatureAnalyzer(sphere, root)
analyzer.analyze(connectivities, scales, intensities, num_runs=1, method="optimization")
sns.histplot([float(c) for c in analyzer.sample_curvatures[0]])
plt.show()
