import warnings

import numpy as np


def get_representative(graphs):
    bin_graphs = []
    for g in graphs:
        g_bin = (g.copy() != 0).astype(int)
        bin_graphs.append(g_bin)
    sum_graph = np.zeros_like(g_bin)
    for g in bin_graphs:
        sum_graph += g
    return (sum_graph == len(graphs)).astype(int)


def compute_distances(graphs, reps):
    distances = np.zeros((len(graphs), len(reps)))

    for i, g in enumerate(graphs):
        for j, r in enumerate(reps):
            b_g = (g != 0).astype(int)
            b_r = (r != 0).astype(int)
            diff = b_r - b_g
            how_many_plus = np.where(diff == -1)[0].size/2
            how_many_less = np.where(diff == 1)[0].size/2
            distances[i, j] = how_many_plus + how_many_less
    return distances


def graph_k_means(graphs, k, max_iter=10):
    ixs = np.arange(0, len(graphs))
    np.random.shuffle(ixs)
    repres = np.array(graphs)[np.array(ixs[:k])]

    labels_prev = [-1]*len(graphs)
    for iter_ in range(max_iter):
        distances = compute_distances(graphs, repres)
        normalized_distances = distances/np.max(distances, axis=1)[:, np.newaxis]
        similarities = 1 - normalized_distances
        kernel = similarities.dot(similarities.T)
        labels = np.argmin(distances, axis=1)
        repres = [get_representative(np.array(graphs)[np.where(labels == v)])
                  for v in np.unique(labels)]
        if np.all(labels == labels_prev):
            break
        labels_prev = labels.copy()
    else:
        warnings.warn("The algorithm did not converge.")
    return kernel
