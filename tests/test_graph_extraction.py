from collections import defaultdict
from typing import List
import unittest, itertools, string
from hypothesis import given
from hypothesis.strategies import lists, sampled_from
from hypothesis.strategies._internal.numbers import integers
import networkx as nx
from networkx.algorithms import isomorphism
from renard.pipeline.graph_extraction import CoOccurencesGraphExtractor
from renard.pipeline.characters_extraction import Character
from renard.pipeline.ner import ner_entities


class TestCoOccurencesGraphExtractor(unittest.TestCase):
    """"""

    # max size used for performance reasons
    @given(lists(sampled_from(string.ascii_uppercase), max_size=7))
    def test_basic_graph_extraction(self, tokens: List[str]):
        bio_tags = ["B-PER" for _ in tokens]

        mentions = ner_entities(tokens, bio_tags)
        characters = defaultdict(list)
        for mention in mentions:
            characters[mention.tokens[0]].append(mention)
        characters = [
            Character(frozenset((name,)), mentions)
            for name, mentions in characters.items()
        ]

        graph_extractor = CoOccurencesGraphExtractor(len(tokens))
        out = graph_extractor(" ".join(tokens), tokens, bio_tags, set(characters))

        G = nx.Graph()
        for i, j in itertools.combinations(range(len(tokens)), 2):
            A, B = (
                Character(
                    frozenset((tokens[i],)),
                    [m for m in mentions if m.tokens[0] == tokens[i]],
                ),
                Character(
                    frozenset((tokens[j],)),
                    [m for m in mentions if m.tokens[0] == tokens[j]],
                ),
            )
            if A == B:
                continue
            if not G.has_edge(A, B):
                G.add_edge(A, B, weight=0)
            G.edges[A, B]["weight"] += 1

        self.assertTrue(
            nx.is_isomorphic(
                out["characters_graph"],
                G,
                edge_match=isomorphism.numerical_edge_match("weight", 0),
            )
        )

    @given(
        lists(sampled_from(string.ascii_uppercase)), integers(min_value=1, max_value=5)
    )
    def test_dynamic_graph_extraction(self, tokens: List[str], dynamic_window: int):
        """
        .. note::

            only tests execution.
        """
        bio_tags = ["B-PER" for _ in tokens]

        mentions = ner_entities(tokens, bio_tags)
        characters = defaultdict(list)
        for mention in mentions:
            characters[mention.tokens[0]].append(mention)
        characters = [
            Character(frozenset((name,)), mentions)
            for name, mentions in characters.items()
        ]

        graph_extractor = CoOccurencesGraphExtractor(
            len(tokens), dynamic="nx", dynamic_window=dynamic_window
        )
        out = graph_extractor(" ".join(tokens), tokens, bio_tags, characters)

        self.assertGreater(len(out["characters_graph"]), 0)


if __name__ == "__main__":
    unittest.main()