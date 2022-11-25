from typing import List, Set
from more_itertools.recipes import flatten
import networkx as nx


def cumulative_graph(graphs: List[nx.Graph]) -> List[nx.Graph]:
    """Turns a dynamic graph to a cumulative graph, weight wise

    :param graphs: A list of sequential graphs
    """
    if len(graphs) == 0:
        return []

    all_attrs = set(flatten([graph_edges_attributes(G) for G in graphs]))

    cumulative_graph = [graphs[0]]
    for H in graphs[1:]:
        G = cumulative_graph[-1]
        # nx.compose creates a new graph with the nodes and edges
        # from both graphs...
        K = nx.compose(H, G)
        # ... however it doesn't sum the attributes : we readjust
        # these here.
        for n1, n2 in K.edges:
            attrs = {}
            for attr in all_attrs:
                G_attr = G.edges.get([n1, n2], default={attr: 0})[attr]
                H_attr = H.edges.get([n1, n2], default={attr: 0})[attr]
                attrs[attr] = G_attr + H_attr
            K.add_edge(n1, n2, **attrs)
        # finally, add the newly created graph to the sequence of
        # cumulative graphs
        cumulative_graph.append(K)

    return cumulative_graph


def graph_edges_attributes(G: nx.Graph) -> Set[str]:
    """Compute the set of all attributes of a graph"""
    return set(flatten(list(data.keys()) for *_, data in G.edges.data()))  # type: ignore
