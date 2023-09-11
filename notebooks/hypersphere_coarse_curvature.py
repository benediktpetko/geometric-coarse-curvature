import numpy as np
import sys

# sys.path.extend(['/home/benedikt/GitHub/geometric-coarse-curvature/venv/lib/python3.9/site-packages'])
# sys.path.extend(['/home/benedikt/GitHub/geometric-coarse-curvature'])

from manifolds.model_spaces import Hypersphere
from analyzers.analyzers import CoarseRicciCurvatureAnalyzer, DisplayMidpointDistances


sphere = Hypersphere()
connectivities = 0.02 / np.arange(1, 2)
scales = 0.3 / np.arange(1, 2)
intensities = 10000 * np.arange(1, 2)
root = np.array([1, 0, 0])

analyzer = CoarseRicciCurvatureAnalyzer(sphere, root)
analyzer.analyze(connectivities, scales, intensities, algorithm="dijkstra", num_runs=1)

DisplayMidpointDistances.plot(analyzer)
