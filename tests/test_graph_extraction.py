from collections import defaultdict
from typing import List
import itertools, string
from hypothesis import given
from hypothesis.strategies import lists, sampled_from
from hypothesis.strategies._internal.numbers import integers
import networkx as nx
from networkx.algorithms import isomorphism
from renard.pipeline.graph_extraction import CoOccurencesGraphExtractor
from renard.pipeline.characters_extraction import Character
from renard.pipeline.ner import ner_entities, NEREntity


def _characters_from_mentions(mentions: List[NEREntity]) -> List[Character]:
    """Generate characters from a list of mentions"""
    name_to_mentions = defaultdict(list)
    for mention in mentions:
        name_to_mentions[mention.tokens[0]].append(mention)
    return [
        Character(frozenset((name,)), mentions)
        for name, mentions in name_to_mentions.items()
    ]


# max size used for performance reasons
@given(lists(sampled_from(string.ascii_uppercase), max_size=7))
def test_basic_graph_extraction(tokens: List[str]):
    bio_tags = ["B-PER" for _ in tokens]

    mentions = ner_entities(tokens, bio_tags)
    characters = _characters_from_mentions(mentions)

    graph_extractor = CoOccurencesGraphExtractor(len(tokens))
    out = graph_extractor(" ".join(tokens), tokens, bio_tags, set(characters), [tokens])

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

    assert nx.is_isomorphic(
        out["characters_graph"],
        G,
        edge_match=isomorphism.numerical_edge_match("weight", 0),
    )


@given(lists(sampled_from(string.ascii_uppercase)), integers(min_value=1, max_value=5))
def test_dynamic_graph_extraction(tokens: List[str], dynamic_window: int):
    """
    .. note::

        only tests execution.
    """
    bio_tags = ["B-PER" for _ in tokens]

    mentions = ner_entities(tokens, bio_tags)
    characters = _characters_from_mentions(mentions)

    graph_extractor = CoOccurencesGraphExtractor(
        len(tokens), dynamic="nx", dynamic_window=dynamic_window
    )
    out = graph_extractor(" ".join(tokens), tokens, bio_tags, set(characters), [tokens])

    assert len(out["characters_graph"]) > 0


@given(lists(sampled_from(string.ascii_uppercase)))
def test_polarity_extraction(tokens: List[str]):
    graph_extractor = CoOccurencesGraphExtractor(10)

    bio_tags = ["B-PER"] * len(tokens)

    mentions = ner_entities(tokens, bio_tags)
    characters = _characters_from_mentions(mentions)

    out = graph_extractor(
        " ".join(tokens),
        tokens,
        bio_tags,
        set(characters),
        sentences=[tokens],
        sentences_polarities=[1.0],
    )

    for character1, character2 in itertools.combinations(characters, 2):
        if out["characters_graph"].has_edge(character1, character2):
            assert "polarity" in out["characters_graph"].edges[character1, character2]


@given(lists(sampled_from(string.ascii_uppercase), min_size=1))
def test_sent_co_occurence_dist(sent1: List[str]):
    # sent2 is guaranteed to be different from sent1, so that we
    # have 2 different characters
    sent2 = [chr(ord(token) + 1) for token in sent1]

    graph_extractor = CoOccurencesGraphExtractor((1, "sentences"))

    sentences = [sent1, sent2]
    tokens = sent1 + sent2
    tags = ["B-PER"] * len(tokens)
    characters = _characters_from_mentions(ner_entities(tokens, tags))

    out = graph_extractor(" ".join(tokens), tokens, tags, set(characters), sentences)

    assert len(out["characters_graph"]) > 0
