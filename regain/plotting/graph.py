import matplotlib.pyplot as plt
import numpy as np


def plot_graph_with_latent_variables(
        adjacency_matrix, n_observed, title="", n_latents=0, labels=None,
        ax=None, node_size=100, font_size=5):
    """Plot graph with latent variables."""
    import networkx as nx
    if ax is None:
        f, ax = plt.subplots(1, 1)

    G = nx.from_numpy_matrix(adjacency_matrix)
    pos = nx.circular_layout(G, scale=.1)

    if n_latents > 0:
        nx.draw_networkx_nodes(
            G, pos, nodelist=list(np.arange(0, n_latents)), ax=ax,
            node_color='r', node_size=node_size, alpha=0.8)

    if labels is not None:
        nx.draw_networkx_labels(G, pos, labels, font_size=font_size, ax=ax)

    nx.draw(
        G, pos, node_color='#A0CBE2', node_size=node_size, edge_color=range(
            len(G.edges.values())), alpha=1, width=2, edge_cmap=plt.cm.Blues,
        with_labels=False, ax=ax)

    ax.set_axis_off()
    ax.set_title(title)
    return ax


def plot_meinshausen_yuan(locations, theta, figsize=(10, 10), file=None):
    plt.figure(figsize=figsize)
    plt.plot(np.array(locations)[:, 0], np.array(locations)[:, 1], 'ro')
    nz = np.nonzero(theta)
    for i, j in zip(nz[0], nz[1]):
        plt.plot(np.array(locations)[[i, j], 0],
                 np.array(locations)[[i, j], 1])
    if file is not None:
        plt.savefig(file, transparency=True, dpi=200, bbox_inches='tight')
    plt.show()
