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

import networkx as nx
import numpy as np
import plotly.graph_objs as go
import seaborn as sns


def _set_ref(x, xref, yref):
    x["xref"] = xref
    x["yref"] = yref
    return x


def _get_idx_interv(d, D):
    k = 0
    while k < len(D) - 1 and d > D[k]:
        k += 1
    return k - 1


def dist(A, B):
    """Euclidean distance between A and B."""
    return np.linalg.norm(np.array(A) - np.array(B))


def _deCasteljau(b, t):
    N = len(b)
    if N < 2:
        raise ValueError("The  control polygon must have at least two points")
    a = np.copy(b)  # shallow copy of the list of control points
    for r in range(1, N):
        a[: N - r, :] = (1 - t) * a[: N - r, :] + t * a[1 : N - r + 1, :]
    return a[0, :]


def _BezierCv(b, nr=5):
    t = np.linspace(0, 1, nr)
    return np.array([_deCasteljau(b, t[k]) for k in range(nr)])


def lines_chord(G, pos, labels, Weights, cmap="Blues"):
    """Create scatter curvy lines based on G."""
    lines = []  # the list of dicts defining   edge  Plotly attributes
    edge_info = []  # the list of points on edges where  the information is placed
    edge_colors = sns.color_palette(cmap, 4).as_hex()
    # ['#d4daff', '#84a9dd', '#5588c8', '#6d8acf']

    Dist = [0, dist([1, 0], 2 * [np.sqrt(2) / 2]), np.sqrt(2), dist([1, 0], [-np.sqrt(2) / 2, np.sqrt(2) / 2]), 2.0]
    params = [1.2, 1.5, 1.8, 2.1]

    for j, e in enumerate(G.edges()):
        A = np.array(pos[e[0]])
        B = np.array(pos[e[1]])
        d = dist(A, B)
        K = _get_idx_interv(d, Dist)
        b = [A, A / params[K], B / params[K], B]
        color = edge_colors[K]
        pts = _BezierCv(b, nr=5)
        text = "{} to {} ({:.2f})".format(labels[e[0]], labels[e[1]], Weights[j])
        mark = _deCasteljau(b, 0.9)
        edge_info.append(
            go.Scatter(
                x=[mark[0]],
                y=[mark[1]],
                mode="markers",
                marker=dict(size=0.5, color=edge_colors),
                text=text,
                hoverinfo="text",
            )
        )
        lines.append(
            go.Scatter(
                x=pts[:, 0],
                y=pts[:, 1],
                mode="lines",
                line=dict(color=color, shape="spline", width=Weights[j] / 5.0),
                hoverinfo="none",
            )
        )

    return lines + edge_info


def lines_straight(G, pos, labels=None, Weights=None, dim=2, **line_params):
    """Create scatter straight lines based on G."""
    # XXX very slow with Weight not None.
    # Waiting for a PR to deal with array of widths
    # https://github.com/plotly/plotly.js/issues/147
    traces = []
    line_pars = dict(x=[], y=[], line=dict(width=0.5, color="#888"), hoverinfo="none", mode="lines")
    line_pars.update(**line_params)

    if Weights is None:
        if dim == 2:
            edge_trace = go.Scatter(**line_pars)

            xs, ys = [], []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                xs.extend([x0, x1, None])
                ys.extend([y0, y1, None])

            edge_trace["x"] = tuple(xs)
            edge_trace["y"] = tuple(ys)
        elif dim == 3:
            edge_trace = go.Scatter3d(z=[], **line_pars)

            xs, ys, zs = [], [], []
            for edge in G.edges():
                x0, y0, z0 = pos[edge[0]]
                x1, y1, z1 = pos[edge[1]]
                xs.extend([x0, x1, None])
                ys.extend([y0, y1, None])
                zs.extend([z0, z1, None])

            edge_trace["x"] = tuple(xs)
            edge_trace["y"] = tuple(ys)
            edge_trace["z"] = tuple(zs)
        else:
            raise ValueError("dim unrecognised")
        traces.append(edge_trace)

    else:
        for j, edge in enumerate(G.edges()):
            if dim == 2:
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
            else:
                x0, y0, z0 = pos[edge[0]]
                x1, y1, z1 = pos[edge[1]]

            Xe = [x0, x1, None]
            Ye = [y0, y1, None]
            if dim == 3:
                Ze = [z0, z1, None]

            scatter_lines = dict(
                x=Xe, y=Ye, mode="lines", line=dict(color="rgb(125,125,125)", width=2 * Weights[j]), hoverinfo="none"
            )
            trace = go.Scatter3d(z=Ze, **scatter_lines) if dim == 3 else go.Scatter(**scatter_lines)
            traces.append(trace)
    return traces


def plot_circular(G, labels=None, dist_text=1.2, cmap="Blues", color_nodes=(), title=""):
    """Plot G as a circular graph.

    Usage
    -----
        import networkx as nx
        G = nx.from_numpy_matrix(A)
        fig = plot_circular(G, labels)
        py.offline.iplot(fig, filename='networkx')
    """
    if labels is None:
        labels = [r"$x_%d$" % i for i in G.nodes]
    pos = nx.circular_layout(G)

    dmin = 1
    # ncenter = 0
    for n in pos:
        x, y = pos[n]
        d = (x - 0.5) ** 2 + (y - 0.5) ** 2
        if d < dmin:
            # ncenter = n
            dmin = d

    # pp = nx.single_source_shortest_path_length(G, ncenter)

    pos_array = np.array(list(pos.values()))  # python 3 compatibility

    center = np.array(
        [
            (np.min(pos_array[:, 0]) + np.max(pos_array[:, 0])) / 2,
            (np.min(pos_array[:, 1]) + np.max(pos_array[:, 1])) / 2,
        ]
    )

    # radius = np.linalg.norm(pos_array - center)
    pos_text = center + dist_text * (pos_array - center)  # text position

    # Compute the text angle
    angles = -180 * np.arctan((pos_array[:, 1] - center[1]) / (pos_array[:, 0] - center[0])) / np.pi

    axis = dict(showgrid=False, zeroline=False, showticklabels=False)
    layout = go.Layout(
        title=title,
        titlefont=dict(size=16),
        showlegend=False,
        hovermode="closest",
        margin=dict(b=40, l=5, r=5, t=40),
        width=650,
        height=650,
        annotations=[
            # dict(
            #     text=
            #     "Python code: <a href='https://plot.ly/ipython-notebooks/network-graphs/'> https://plot.ly/ipython-notebooks/network-graphs/</a>",
            #     showarrow=False, xref="paper", yref="paper", x=0.005, y=-0.002)
        ],
        xaxis=dict(axis),
        yaxis=dict(axis),
    )

    # annotations around the circle
    # textangle=angles[k] is the angle with horizontal line through (x,y)
    # in degrees
    # + =clockwise, -=anti-clockwise
    layout["annotations"] += tuple(
        [
            go.layout.Annotation(
                x=pos_text[k][0],
                y=pos_text[k][1],
                text=labels[k],
                textangle=angles[k],
                font=dict(size=20, color="rgb(0,0,0)"),
                showarrow=False,
            )
            for k in range(pos_array.shape[0])
        ]
    )

    # V = list(G.nodes())

    Es = G.edges()
    Weights = [edge["weight"] for edge in Es.values()]

    edge_trace = lines_chord(G, pos, labels, Weights, cmap=cmap)
    node_trace = go.Scatter(
        x=[],
        y=[],
        text=[],
        mode="markers",
        hoverinfo="text",
        marker=dict(
            showscale=False,
            colorscale=cmap,
            reversescale=True,
            color=color_nodes,
            size=20,
            colorbar=dict(thickness=15, title="Node Connections", xanchor="left", titleside="right"),
            line=dict(width=2),
        ),
    )

    for node in G.nodes():
        x, y = pos[node]
        node_trace["x"] += tuple([x])
        node_trace["y"] += tuple([y])

    for node, adjacencies in enumerate(G.adjacency()):
        node_trace["marker"]["color"] += tuple([len(adjacencies[1])])
        node_info = "# of connections: {}".format(len(adjacencies[1]))
        node_trace["text"] += tuple([node_info])

    data = edge_trace + [node_trace]
    return go.Figure(data=data, layout=layout)


def plot_network(G, labels, with_weights=True, dim=2, title=None, cmap="Blues", **line_params):
    """Plot network with plotly."""
    pos = nx.fruchterman_reingold_layout(G, dim=dim)
    pos_vals = np.array([pos[k] for k in pos]).T
    if dim == 2:
        Xn, Yn = pos_vals
    elif dim == 3:
        Xn, Yn, Zn = pos_vals

    weights = [edge["weight"] for edge in G.edges.values()] if with_weights else None
    degrees = np.array(G.degree)[:, 1]
    traces = lines_straight(G, pos, Weights=weights, dim=dim, **line_params)
    marker = dict(
        showscale=False,
        colorscale=cmap,
        reversescale=True,
        color=degrees,
        size=2,
        colorbar=dict(thickness=15, title="Node Connections", xanchor="left", titleside="right"),
        line=dict(width=1),
    )

    text_node = ["{}<br># of connections: {}".format(lbl, info) for lbl, info in zip(labels, degrees)]
    node_trace_params = dict(x=Xn, y=Yn, mode="markers", marker=marker, text=text_node, hoverinfo="text")
    if dim == 3:
        # fix problem with 3D plot hover
        # see https://github.com/plotly/plotly.py/issues/952
        node_traces = [
            go.Scatter3d(
                x=[0], y=[0], z=[0], marker={"color": "rgb(0, 0, 0)", "opacity": 1, "size": 0.1}, showlegend=False
            )
        ]

        node_traces += [go.Scatter3d(z=Zn, **node_trace_params)]
    else:
        node_traces = [
            go.Scatter(
                **dict(
                    node_trace_params,
                    marker=dict(marker, symbol="circle-dot"),
                )
            )
        ]

    axis = dict(showline=False, zeroline=False, showgrid=False, showticklabels=False, title="")

    lay = dict(title=title, width=500, height=500, showlegend=False, margin=dict(t=100), hovermode="closest")
    if dim == 2:
        layout = go.Layout(
            **dict(
                lay,
                scene=dict(xaxis=dict(axis), yaxis=dict(axis), zaxis=dict(axis)),
                xaxis=dict(axis),
                yaxis=dict(axis),
            )
        )
    else:
        layout = go.Layout(
            **dict(
                lay,
                scene=dict(
                    xaxis=dict(axis, showbackground=False),
                    yaxis=dict(axis, showbackground=False),
                    zaxis=dict(axis, showbackground=False),
                ),
                xaxis=dict(axis),
                yaxis=dict(axis),
            )
        )

    return go.Figure(data=node_traces + traces, layout=layout)
