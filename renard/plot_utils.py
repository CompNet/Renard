import math
from typing import TYPE_CHECKING, Any, Dict, Optional, Union, Tuple
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from renard.graph_utils import graph_edges_attributes

if TYPE_CHECKING:
    from renard.pipeline.character_unification import Character


CharactersGraphLayout = Union[
    Dict["Character", Tuple[float, float]], Dict["Character", np.ndarray]
]


def layout_nx_graph_reasonably(G: nx.Graph) -> Dict[Any, np.ndarray]:
    return nx.spring_layout(G, k=min(1.5, 8 / math.sqrt(len(G.nodes))))  # type: ignore


def plot_nx_graph_reasonably(
    G: nx.Graph,
    ax=None,
    layout: Optional[dict] = None,
    node_kwargs: Optional[Dict[str, Any]] = None,
    edge_kwargs: Optional[Dict[str, Any]] = None,
    label_kwargs: Optional[Dict[str, Any]] = None,
    legend: bool = False,
):
    """Try to plot a :class:`nx.Graph` with 'reasonable' parameters

    :param G: the graph to draw
    :param ax: matplotlib axes
    :param layout: if given, this graph layout will be applied.
        Otherwise, use :func:`layout_nx_graph_reasonably`.
    :param node_kwargs: passed to :func:`nx.draw_networkx_nodes`
    :param edge_kwargs: passed to :func:`nx.draw_networkx_nodes`
    :param label_kwargs: passed to :func:`nx.draw_networkx_labels`
    :param legend: if ``True``, will try to plot an additional legend.
    """
    pos = layout
    if pos is None:
        pos = layout_nx_graph_reasonably(G)

    node_kwargs = node_kwargs or {}
    node_kwargs["node_color"] = node_kwargs.get(
        "node_color", [degree for _, degree in G.degree]
    )
    node_kwargs["cmap"] = node_kwargs.get("cmap", "viridis")
    node_kwargs["node_size"] = node_kwargs.get(
        "node_size", [1 + degree * 10 for _, degree in G.degree]
    )
    scatter = nx.draw_networkx_nodes(G, pos, ax=ax, **node_kwargs)
    if legend:
        if ax:
            ax.legend(*scatter.legend_elements("sizes"))
        else:
            plt.legend(*scatter.legend_elements("sizes"))

    edge_kwargs = edge_kwargs or {}
    edges_attrs = graph_edges_attributes(G)
    if (
        not "edge_color" in edge_kwargs
        and not "edge_cmap" in edge_kwargs
        and "polarity" in edges_attrs
    ):
        # we draw the polarity of interactions if the 'polarity'
        # attribute is present in the graph
        polarities = [d.get("polarity", 0) for *_, d in G.edges.data()]  # type: ignore
        edge_kwargs["edge_color"] = ["g" if p > 0 else "r" for p in polarities]
        edge_kwargs["edge_cmap"] = None
    else:
        edge_kwargs["edge_color"] = edge_kwargs.get(
            "edge_color", [math.log(d.get("weight", 1)) for *_, d in G.edges.data()]
        )
        edge_kwargs["edge_cmap"] = edge_kwargs.get("edge_cmap", plt.get_cmap("viridis"))
    edge_kwargs["width"] = edge_kwargs.get(
        "width", [1 + math.log(d.get("weight", 1)) for _, _, d in G.edges.data()]
    )
    edge_kwargs["alpha"] = edge_kwargs.get("alpha", 0.35)
    nx.draw_networkx_edges(G, pos, ax=ax, **edge_kwargs)

    label_kwargs = label_kwargs or {}
    label_kwargs["verticalalignment"] = label_kwargs.get("verticalalignment", "top")
    label_kwargs["font_size"] = label_kwargs.get("font_size", 8)
    label_kwargs["alpha"] = label_kwargs.get("alpha", 0.75)
    nx.draw_networkx_labels(G, pos=pos, ax=ax, **label_kwargs)
