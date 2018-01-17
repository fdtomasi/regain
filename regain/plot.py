"""Plotting utils."""
import matplotlib.pyplot as plt
import numpy as np


def plot_graph_with_latent_variables(adjacency_matrix, n_latents, n_observed,
                                     title):
    """Plot graph with latent variables."""
    import networkx as nx
    # plt.figure(figsize=(15,10))
    plt.figure()
    G = nx.from_numpy_matrix(adjacency_matrix)
    pos = nx.circular_layout(G)

    nx.draw_networkx_nodes(
        G, pos, nodelist=list(np.arange(0, n_latents)),
        node_color='r', node_size=500, alpha=0.8)
    nx.draw_networkx_nodes(
        G, pos, nodelist=list(np.arange(n_latents, n_latents + n_observed)),
        node_color='b', node_size=500, alpha=0.8)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    plt.axis("off")
    plt.title(title)
    plt.show()
