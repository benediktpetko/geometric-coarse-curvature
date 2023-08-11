import matplotlib.pyplot as plt
import numpy as np

from structures.geometric_graph import GeometricGraph


def curvature_convergence_analyzer(
        manifold, root, connectivities, scales, intensities, num_runs=1,
        method="triangular", algorithm='floyd-warshall'):
    """
    Compare the curvature at the root of a geometric graph to the curvature of the manifold
    at the corresponding point.

    :param algorithm:
    :param manifold: the manifold on which to sample points
    :param root: the root of the geometric graph
    :param connectivities: array of the connectivities of the geometric graph
    :param scales: array of the scales of the coarse curvature
    :param intensities: array of intensities of the Poisson point process
    :param num_runs: the number of times to run the experiment for averaging
    :param method: the method to use to compute the coarse curvature
    :return: a list of average curvature at the root of the graph
    """
    results = []

    for i in range(len(intensities)):
        print("Point density factor: ", intensities[i] * connectivities[i] ** manifold.dim)
        print("Geodesic approximation factor: ", scales[i] / connectivities[i])
        sample_curvatures = []
        for _ in range(num_runs):
            point_cloud = manifold.poisson_sample(intensities[i])
            geometric_graph = GeometricGraph(point_cloud, root, connectivities[i])
            try:
                ricci_curvature = 2 * (manifold.dim + 2) / (scales[i] ** 2) * \
                              geometric_graph.compute_coarse_curvature(scales[i], method=method, algorithm=algorithm)
            except IndexError:
                continue
            print("Estimated Ricci curvature: ", ricci_curvature, "\n")
            sample_curvatures.append(ricci_curvature)
        result = sum(sample_curvatures) / len(sample_curvatures)
        results.append(result)
        print(f"Scale: {scales[i]}, curvature: {result} \n")
        print(f"Estimate from {len(sample_curvatures)} samples: {result} \n")
    plt.plot(range(len(intensities)), results)
    plt.show()
    print("Curvatures at root: \n", results)
