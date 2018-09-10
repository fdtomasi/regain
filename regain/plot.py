"""Plotting utils."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse
from scipy import linalg


def plot_graph_with_latent_variables(
        adjacency_matrix, n_observed, title="", n_latents=0,
        labels=None, ax=None, node_size=100, font_size=5):
    """Plot graph with latent variables."""
    import networkx as nx
    # plt.figure(figsize=(15,10))
    if ax is None:
        f, ax = plt.subplots(1,1)

    G = nx.from_numpy_matrix(adjacency_matrix)
    pos = nx.circular_layout(G, scale=.1)

    if n_latents > 0:
        nx.draw_networkx_nodes(
            G, pos, nodelist=list(np.arange(0, n_latents)), ax=ax,
            node_color='r', node_size=node_size, alpha=0.8)

    #nx.draw_networkx_nodes(
    #    G, pos, nodelist=list(np.arange(n_latents, n_latents + n_observed)), ax=ax,
    #    node_color='b', node_size=node_size, alpha=0.2)
    if labels is not None:
        nx.draw_networkx_labels(G, pos, labels, font_size=font_size, ax=ax)

    #nx.draw_networkx_edges(
    #    G, pos, colors=range(n_observed), cmap=plt.cm.Blues,width=1.0, alpha=0.5, ax=ax)
    nx.draw(G, pos, node_color='#A0CBE2', node_size=node_size,
            edge_color=range(len(G.edges.values())), alpha=1,
            width=2, edge_cmap=plt.cm.Blues, with_labels=False, ax=ax)

    ax.set_axis_off()
    ax.set_title(title)
    return ax


def plot_cov_2d(means, cov, sdwidth=1.0, npts=50, ax=None, c=None):
    tt = np.linspace(0, 2 * np.pi, 100)
    ap = np.array([np.cos(tt), np.sin(tt)])

    d, v = linalg.eigh(cov)
    d = sdwidth * np.sqrt(d)

    bp = (v.dot(np.diag(d)).dot(ap)) + means.ravel()[:, None]
    plot = plt.plot if ax is None else ax.plot
    plot(bp[0], bp[1], c=c)


def plot_cov_ellipse(pos, cov, nstd=2, ax=None, **kwargs):
    """
    Plots an `nstd` sigma error ellipse based on the specified covariance
    matrix (`cov`). Additional keyword arguments are passed on to the
    ellipse patch artist.
    Parameters
    ----------
        cov : The 2x2 covariance matrix to base the ellipse on
        pos : The location of the center of the ellipse. Expects a 2-element
            sequence of [x0, y0].
        nstd : The radius of the ellipse in numbers of standard deviations.
            Defaults to 2 standard deviations.
        ax : The axis that the ellipse will be plotted on. Defaults to the
            current axis.
        Additional keyword arguments are pass on to the ellipse patch.
    Returns
    -------
        A matplotlib ellipse artist
    """

    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:, order]

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)

    if width == 0:
        width += 0.1
    if height == 0:
        height += 0.1

    if ax is None:
        ax = plt.gca()
        plt.xlim([- (width + pos[0] + 0.1), width + pos[0] + 0.1])
        plt.ylim([- (height + pos[1] + 0.1), height + pos[1] + 0.1])
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip
