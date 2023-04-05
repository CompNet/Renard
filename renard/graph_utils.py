from __future__ import annotations
from typing import List, Set, Union, Literal, Callable, TYPE_CHECKING
from more_itertools.recipes import flatten
import networkx as nx

if TYPE_CHECKING:
    from renard.pipeline.characters_extraction import Character


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
        # We also re-add the graph and nodes attributes from G
        K.graph = H.graph
        # finally, add the newly created graph to the sequence of
        # cumulative graphs
        cumulative_graph.append(K)

    return cumulative_graph


def graph_edges_attributes(G: nx.Graph) -> Set[str]:
    """Compute the set of all attributes of a graph"""
    return set(flatten(list(data.keys()) for *_, data in G.edges.data()))  # type: ignore


def graph_with_names(
    G: nx.Graph,
    name_style: Union[
        Literal["longest", "shortest", "most_frequent"], Callable[[Character], str]
    ] = "longest",
) -> nx.Graph:
    """Relabel a characters graph, using a single name for each
    node

    :param name_style: characters name style in the resulting
        graph.  Either a string (``'longest`` or ``shortest`` or
        ``most_frequent``) or a custom function associating a
        character to its name
    """
    if name_style == "longest":
        name_style_fn = lambda character: character.longest_name()
    elif name_style == "shortest":
        name_style_fn = lambda character: character.shortest_name()
    elif name_style == "most_frequent":
        name_style_fn = lambda character: character.most_frequent_name()
    else:
        name_style_fn = name_style

    return nx.relabel_nodes(
        G,
        {character: name_style_fn(character) for character in G.nodes()},  # type: ignore
    )
