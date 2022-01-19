from typing import Any, Dict, List, FrozenSet, Set, Optional
from itertools import combinations
from collections import defaultdict
from dataclasses import dataclass
from renard.gender import Gender
from renard.pipeline.ner import bio_entities
from renard.pipeline.core import PipelineStep
from renard.resources.hypocorisms import HypocorismGazetteer
from renard.resources.pronouns.pronouns import is_a_female_pronoun, is_a_male_pronoun


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
            if count >= self.min_appearance
        ]

        return {"characters": characters}

    def needs(self) -> Set[str]:
        return {"tokens", "bio_tags"}

    def production(self) -> Set[str]:
        return {"characters"}


class GraphRulesCharactersExtractor(PipelineStep):
    """Extract characters by creating a graph where mentions are
    linked when they refer to the same character, and then
    merging this graph nodes.

    This algorithm is inspired from Vala et al., 2015

    .. warning::

        This is a work in progress.

    """

    def __init__(self) -> None:
        self.hypocorism_gazetteer = HypocorismGazetteer()

    def __call__(
        self, tokens: List[str], bio_tags: List[str], **kwargs: Any
    ) -> Dict[str, Any]:
        assert len(tokens) == len(bio_tags)

        corefs: Optional[List[List[dict]]] = None
        if "corefs" in kwargs:
            corefs = kwargs["corefs"]

        import networkx as nx

        # create a graph where each node is a mention detected by NER
        G = nx.Graph()
        for mention, tag, token_idx in bio_entities(tokens, bio_tags):
            if tag.startswith("PER"):
                G.add_node(mention)

        # rules-based links
        for (name1, name2) in combinations(G.nodes(), 2):
            if (
                self.hypocorism_gazetteer.are_related(name1, name2)
                or self.names_are_related_after_title_removal(name1, name2)
                or (
                    not corefs is None and self.names_are_in_coref(name1, name2, corefs)
                )
            ):
                G.add_edge(name1, name2)

        # unlink rules
        for (name1, name2) in combinations(G.nodes(), 2):
            if corefs is None:
                break
            # if characters dont have the same infered gender,
            # unlink the shortes path between them
            gender1 = self.infer_name_gender(name1, corefs)
            gender2 = self.infer_name_gender(name2, corefs)
            if gender1 != gender2:
                for path in nx.all_shortest_paths(G, source=name1, target=name2):
                    G.remove_edges_from(path)

        return {
            "characters": [
                Character(frozenset(names)) for names in nx.connected_components(G)
            ]
        }

    def names_are_related_after_title_removal(self, name1: str, name2: str) -> bool:
        from nameparser import HumanName
        from nameparser.config import CONSTANTS

        CONSTANTS.string_format = "{first} {middle} {last}"
        raw_name1 = HumanName(name1).full_name
        raw_name2 = HumanName(name2).full_name

        return raw_name1 == raw_name2 or self.hypocorism_gazetteer.are_related(
            raw_name1, raw_name2
        )

    def names_are_in_coref(self, name1: str, name2: str, corefs: List[List[dict]]):
        for coref_chain in corefs:
            if any([name1 == m["mention"] for m in coref_chain]) and any(
                [name2 == m["mention"] for m in coref_chain]
            ):
                return True
        return False

    def infer_name_gender(self, name: str, corefs: List[List[dict]]) -> Gender:
        """Try to infer a character's gender from coreferences chains"""

        female_count = 0
        male_count = 0

        for coref_chain in corefs:
            mentions = {m["mention"] for m in coref_chain}
            if not name in mentions:
                continue
            for mention in mentions:
                if is_a_male_pronoun(mention):
                    male_count += 1
                elif is_a_female_pronoun(mention):
                    female_count += 1

        if male_count == female_count:
            return Gender.UNKNOWN
        return Gender.MALE if male_count > female_count else Gender.FEMALE

    def needs(self) -> Set[str]:
        return {"tokens", "bio_tags"}

    def optional_needs(self) -> Set[str]:
        return {"corefs"}

    def production(self) -> Set[str]:
        return {"characters"}
