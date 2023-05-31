import math
from typing import Any, Dict, Optional
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from renard.graph_utils import graph_edges_attributes


def layout_nx_graph_reasonably(G: nx.Graph) -> Dict[Any, np.ndarray]:
    return nx.spring_layout(G, k=2 / math.sqrt(len(G.nodes)))  # type: ignore


def plot_nx_graph_reasonably(
    G: nx.Graph, ax=None, layout: Optional[Dict[Any, np.ndarray]] = None
):
    """Try to plot a :class:`nx.Graph` with 'reasonable' parameters

    :param G: the graph to draw
    :param ax: matplotlib axes
    :param layout: if given, this graph layout will be applied.
        Otherwise, use :func:`layout_nx_graph_reasonably`.
    """
    pos = layout
    if pos is None:
        pos = layout_nx_graph_reasonably(G)

    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=[degree for _, degree in G.degree],  # type: ignore
        cmap=plt.get_cmap("winter_r"),
        node_size=[degree * 10 for _, degree in G.degree],  # type: ignore
        ax=ax,
    )

    edges_attrs = graph_edges_attributes(G)
    if "polarity" in edges_attrs:
        # we draw the polarity of interactions if the 'polarity'
        # attribute is present in the graph
        polarities = [d.get("polarity", 0) for *_, d in G.edges.data()]  # type: ignore
        edge_color = ["g" if p > 0 else "r" for p in polarities]
        edge_cmap = None

    else:
        edge_color = [math.log(d["weight"]) for *_, d in G.edges.data()]
        edge_cmap = plt.get_cmap("winter_r")
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=edge_color,
        edge_cmap=edge_cmap,
        edge_vmax=1,
        edge_vmin=-1,
        width=[1 + math.log(d["weight"]) for _, _, d in G.edges.data()],  # type: ignore
        alpha=0.35,
        ax=ax,
    )

    nx.draw_networkx_labels(
        G, pos=pos, ax=ax, verticalalignment="top", font_size=8, alpha=0.75
    )
