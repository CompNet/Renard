import math
from typing import Any, Dict, Optional
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt


def layout_nx_graph_reasonably(G: nx.Graph) -> Dict[Any, np.ndarray]:
    return nx.spring_layout(G, k=0.75 * math.sqrt(len(G.nodes)))  # type: ignore


def draw_nx_graph_reasonably(
    G: nx.Graph, ax=None, layout: Optional[Dict[Any, np.ndarray]] = None
):
    """Try to draw a :class:`nx.Graph` with 'reasonable' parameters

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
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=[math.log(d["weight"]) for _, _, d in G.edges.data()],  # type: ignore
        edge_cmap=plt.get_cmap("winter_r"),
        width=[1 + math.log(d["weight"]) for _, _, d in G.edges.data()],  # type: ignore
        alpha=0.35,
        ax=ax,
    )
    nx.draw_networkx_labels(G, pos=pos, ax=ax, verticalalignment="top")
