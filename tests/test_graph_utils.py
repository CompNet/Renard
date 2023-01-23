import networkx as nx
from renard.graph_utils import cumulative_graph


def test_cumulative_graph():
    gs = [
        nx.Graph([(0, 1, {"weight": 1})]),
        nx.Graph([(0, 1, {"weight": 1}), (0, 2, {"weight": 1})]),
    ]

    assert nx.is_isomorphic(
        cumulative_graph(gs)[-1],
        nx.Graph([(0, 1, {"weight": 2}), (0, 2, {"weight": 1})]),
    )
