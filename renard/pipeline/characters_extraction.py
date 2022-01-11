from renard.pipeline.ner import bio_entities
from typing import Any, Dict, List, FrozenSet, Set
from collections import defaultdict
from dataclasses import dataclass
from renard.pipeline.core import PipelineStep
from renard.resources.hypocorisms import HypocorismGazetteer


@dataclass(eq=True, frozen=True)
class Character:
    names: FrozenSet[str]

    def longest_name(self) -> str:
        return max(self.names, key=len)


class NaiveCharactersExtractor(PipelineStep):
    """A basic character extractor using NER"""

    def __init__(self, min_appearance: int) -> None:
        """
        :param min_appearance:
        """
        self.min_appearance = min_appearance

    def __call__(
        self, text: str, tokens: List[str], bio_tags: List[str], **kwargs
    ) -> Dict[str, Any]:
        """
        :param text:
        :param tokens:
        :param bio_tags:
        """
        assert len(tokens) == len(bio_tags)

        entities = defaultdict(int)
        current_entity: List[str] = []
        for token, tag in zip(tokens, bio_tags):
            if len(current_entity) == 0:
                if tag.startswith("B-PER"):
                    current_entity.append(token)
            else:
                if tag.startswith("I-PER"):
                    current_entity.append(token)
                else:  # end of entity
                    entities[" ".join(current_entity)] += 1
                    current_entity = []

        characters = [
            Character(frozenset((entity,)))
            for entity, count in entities.items()
            if count > self.min_appearance
        ]

        return {"characters": characters}

    def needs(self) -> Set[str]:
        return {"tokens", "bio_tags"}

    def produces(self) -> Set[str]:
        return {"characters"}


class GraphRulesCharactersExtractor(PipelineStep):
    """Extract characters by creating a graph where mentions are
    linked when they refer to the same character, and then
    merging this graph nodes.

    This algorithm is inspired from Vala et al., 2015
    """

    def __init__(self) -> None:
        self.hypocorism_gazetteer = HypocorismGazetteer()

    def __call__(
        self, tokens: List[str], bio_tags: List[str], **kwargs: Any
    ) -> Dict[str, Any]:
        assert len(tokens) == len(bio_tags)

        import networkx as nx

        # create a graph where each node is a mention detected by NER
        G = nx.Graph()
        for mention, tag, token_idx in bio_entities(tokens, bio_tags):
            if tag.startswith("PER"):
                G.add_node(mention)

        # links using nickname gazetteer
        for name1 in G:
            for name2 in G:
                if name1 == name2:
                    continue
                if self.hypocorism_gazetteer.are_related(name1, name2):
                    G.add_edge(name1, name2)

        return {
            "characters": [Character(names) for names in nx.connected_components(G)]
        }

    def needs(self) -> Set[str]:
        return {"tokens", "bio_tags"}

    def produces(self) -> Set[str]:
        return {"characters"}
