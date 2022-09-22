import math
import networkx as nx
import matplotlib.pyplot as plt


def draw_nx_graph_reasonably(G: nx.Graph, ax=None):
    """Try to draw a :class:`nx.Graph` with 'reasonable' parameters

    :param G: the graph to draw
    :param ax: matplotlib axes
    """
    pos = nx.spring_layout(G, k=0.5 * math.sqrt(len(G.nodes)))
    nx.draw_networkx_nodes(
        G,
        pos,
        node_color=[degree for _, degree in G.degree],
        cmap=plt.get_cmap("winter_r"),
        node_size=[degree * 10 for _, degree in G.degree],
        ax=ax,
    )
    nx.draw_networkx_edges(
        G,
        pos,
        edge_color=[math.log(d["weight"]) for _, _, d in G.edges.data()],  # type: ignore
        edge_cmap=plt.get_cmap("winter_r"),
        width=[1 + math.log(d["weight"]) for _, _, d in G.edges.data()],
        alpha=0.35,
        ax=ax,
    )
    nx.draw_networkx_labels(G, pos=pos, ax=ax, verticalalignment="top")
