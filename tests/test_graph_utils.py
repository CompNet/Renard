import unittest
import networkx as nx
from renard.graph_utils import cumulative_graph


class TestCumulativeGraph(unittest.TestCase):
    """"""

    def test_cumulative_graph(self):
        gs = [
            nx.Graph([(0, 1, {"weight": 1})]),
            nx.Graph([(0, 1, {"weight": 1}), (0, 2, {"weight": 1})]),
        ]

        self.assertTrue(
            nx.is_isomorphic(
                cumulative_graph(gs)[-1],
                nx.Graph([(0, 1, {"weight": 2}), (0, 2, {"weight": 1})]),
            )
        )
