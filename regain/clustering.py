# BSD 3-Clause License

# Copyright (c) 2019, regain authors
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

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
            how_many_plus = np.where(diff == -1)[0].size / 2
            how_many_less = np.where(diff == 1)[0].size / 2
            distances[i, j] = how_many_plus + how_many_less
    return distances


def graph_k_means(graphs, k, max_iter=10):
    ixs = np.arange(0, len(graphs))
    np.random.shuffle(ixs)
    repres = np.array(graphs)[np.array(ixs[:k])]

    labels_prev = [-1] * len(graphs)
    for iter_ in range(max_iter):
        distances = compute_distances(graphs, repres)
        normalized_distances = distances / np.max(distances, axis=1)[:, np.newaxis]
        similarities = 1 - normalized_distances
        kernel = similarities.dot(similarities.T)
        labels = np.argmin(distances, axis=1)
        repres = [get_representative(np.array(graphs)[np.where(labels == v)]) for v in np.unique(labels)]
        if np.all(labels == labels_prev):
            break
        labels_prev = labels.copy()
    else:
        warnings.warn("The algorithm did not converge.")
    return kernel
