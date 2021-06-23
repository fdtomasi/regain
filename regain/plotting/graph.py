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
"""Utilities to plot a graph."""
import matplotlib.pyplot as plt
import numpy as np


def plot_graph_with_latent_variables(
    adjacency_matrix, n_observed, title="", n_latents=0, labels=None, ax=None, node_size=100, font_size=5
):
    """Plot graph with latent variables."""
    import networkx as nx

    if ax is None:
        f, ax = plt.subplots(1, 1)

    G = nx.from_numpy_matrix(adjacency_matrix)
    pos = nx.circular_layout(G, scale=0.1)

    if n_latents > 0:
        nx.draw_networkx_nodes(
            G, pos, nodelist=list(np.arange(0, n_latents)), ax=ax, node_color="r", node_size=node_size, alpha=0.8
        )

    if labels is not None:
        nx.draw_networkx_labels(G, pos, labels, font_size=font_size, ax=ax)

    nx.draw(
        G,
        pos,
        node_color="#A0CBE2",
        node_size=node_size,
        edge_color=range(len(G.edges.values())),
        alpha=1,
        width=2,
        edge_cmap=plt.cm.Blues,
        with_labels=False,
        ax=ax,
    )

    ax.set_axis_off()
    ax.set_title(title)
    return ax


def plot_meinshausen_yuan(locations, theta, figsize=(10, 10), file=None):
    """Plot graph with latent variables."""
    plt.figure(figsize=figsize)
    plt.plot(np.array(locations)[:, 0], np.array(locations)[:, 1], "ro")
    nz = np.nonzero(theta)
    for i, j in zip(nz[0], nz[1]):
        plt.plot(np.array(locations)[[i, j], 0], np.array(locations)[[i, j], 1])
    if file is not None:
        plt.savefig(file, transparency=True, dpi=200, bbox_inches="tight")
    plt.show()
