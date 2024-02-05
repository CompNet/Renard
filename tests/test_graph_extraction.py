from collections import defaultdict
from typing import List
import itertools, string
from hypothesis import given
from hypothesis.strategies import lists, sampled_from
from hypothesis.strategies._internal.numbers import integers
import networkx as nx
from networkx.algorithms import isomorphism
from renard.pipeline.graph_extraction import CoOccurrencesGraphExtractor
from renard.pipeline.character_unification import Character
from renard.pipeline.ner import ner_entities, NEREntity


def _characters_from_mentions(mentions: List[NEREntity]) -> List[Character]:
    """Generate characters from a list of mentions"""
    name_to_mentions = defaultdict(list)
    for mention in mentions:
        name_to_mentions[" ".join(mention.tokens)].append(mention)
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

    graph_extractor = CoOccurrencesGraphExtractor(len(tokens))
    out = graph_extractor(set(characters), [tokens])

    characters = {
        token: Character(
            frozenset([token]), [m for m in mentions if m.tokens[0] == token]
        )
        for token in set(tokens)
    }

    G = nx.Graph()
    for character in characters.values():
        G.add_node(character)

    for i, j in itertools.combinations(range(len(tokens)), 2):
        A = characters[tokens[i]]
        B = characters[tokens[j]]
        if A == B:
            continue
        if not G.has_edge(A, B):
            G.add_edge(A, B, weight=0)
        G.edges[A, B]["weight"] += 1

    assert nx.is_isomorphic(
        out["character_network"],
        G,
        edge_match=isomorphism.numerical_edge_match("weight", 0),
    )


@given(
    lists(sampled_from(string.ascii_uppercase), min_size=1),
    integers(min_value=1, max_value=5),
)
def test_dynamic_graph_extraction(tokens: List[str], dynamic_window: int):
    """
    .. note::

        only tests execution.
    """
    bio_tags = ["B-PER" for _ in tokens]

    mentions = ner_entities(tokens, bio_tags)
    characters = _characters_from_mentions(mentions)

    graph_extractor = CoOccurrencesGraphExtractor(
        len(tokens), dynamic=True, dynamic_window=dynamic_window
    )
    out = graph_extractor(set(characters), [tokens])

    assert len(out["character_network"]) > 0


@given(lists(sampled_from(string.ascii_uppercase)))
def test_polarity_extraction(tokens: List[str]):
    graph_extractor = CoOccurrencesGraphExtractor(10)

    bio_tags = ["B-PER"] * len(tokens)

    mentions = ner_entities(tokens, bio_tags)
    characters = _characters_from_mentions(mentions)

    out = graph_extractor(
        set(characters),
        sentences=[tokens],
        sentences_polarities=[1.0],
    )

    for character1, character2 in itertools.combinations(characters, 2):
        if out["character_network"].has_edge(character1, character2):
            assert "polarity" in out["character_network"].edges[character1, character2]


@given(lists(sampled_from(string.ascii_uppercase), min_size=1))
def test_sent_co_occurence_dist(sent1: List[str]):
    # sent2 is guaranteed to be different from sent1, so that we
    # have 2 different characters
    sent2 = [chr(ord(token) + 1) for token in sent1]

    graph_extractor = CoOccurrencesGraphExtractor((1, "sentences"))

    sentences = [sent1, sent2]
    tokens = sent1 + sent2
    tags = ["B-PER"] * len(tokens)
    characters = _characters_from_mentions(ner_entities(tokens, tags))

    out = graph_extractor(set(characters), sentences)

    assert len(out["character_network"]) > 0
