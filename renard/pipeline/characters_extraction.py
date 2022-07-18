import re
from typing import Any, Dict, List, FrozenSet, Set, Optional, Tuple
from itertools import combinations
from collections import Counter
from dataclasses import dataclass

from nameparser import config
from nameparser import HumanName

from renard.gender import Gender
from renard.pipeline.ner import ner_entities
from renard.pipeline.core import PipelineStep
from renard.pipeline.corefs import CoreferenceChain
from renard.resources.hypocorisms import HypocorismGazetteer
from renard.resources.pronouns.pronouns import is_a_female_pronoun, is_a_male_pronoun


@dataclass(eq=True, frozen=True)
class Character:
    names: FrozenSet[str]

    def longest_name(self) -> str:
        return max(self.names, key=len)

    def shortest_name(self) -> str:
        return min(self.names, key=len)

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.names)))


class NaiveCharactersExtractor(PipelineStep):
    """A basic character extractor using NER"""

    def __init__(self, min_appearances: int = 0) -> None:
        """
        :param min_appearances: minimum number of appearances of a
            character for it to be extracted
        """
        self.min_appearances = min_appearances
        super().__init__()

    def __call__(
        self, text: str, tokens: List[str], bio_tags: List[str], **kwargs
    ) -> Dict[str, Any]:
        """
        :param text:
        :param tokens:
        :param bio_tags:
        """
        assert len(tokens) == len(bio_tags)

        entities = ner_entities(tokens, bio_tags)
        entities_c = Counter([" ".join(e.tokens) for e in entities])
        entities = [e for e, c in entities_c.items() if c >= self.min_appearances]

        characters = [Character(frozenset((entity,))) for entity in entities]

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

    def __init__(
        self,
        min_appearances: int = 0,
        additional_hypocorisms: Optional[List[Tuple[str, List[str]]]] = None,
    ) -> None:
        """
        :param min_appearances: minimum number of appearances of a
            character for it to be extracted.
        :param additional_hypocorisms: a tuple of additional
            hypocorisms.  Each hypocorism is a tuple where the first
            element is a name, and the second element is a set of
            nicknames associated with it
        """
        self.min_appearances = min_appearances

        self.hypocorism_gazetteer = HypocorismGazetteer()
        if not additional_hypocorisms is None:
            for name, nicknames in additional_hypocorisms:
                self.hypocorism_gazetteer._add_hypocorism_(name, nicknames)

        super().__init__()

    def __call__(
        self,
        tokens: List[str],
        bio_tags: List[str],
        corefs: Optional[List[CoreferenceChain]] = None,
        **kwargs: dict
    ) -> Dict[str, Any]:
        assert len(tokens) == len(bio_tags)

        import networkx as nx

        occurences = [" ".join(e.tokens) for e in ner_entities(tokens, bio_tags)]

        # create a graph where each node is a mention detected by NER
        G = nx.Graph()
        for occurence in occurences:
            G.add_node(occurence)

        # link nodes based on several rules
        for (name1, name2) in combinations(G.nodes(), 2):
            if (
                self.hypocorism_gazetteer.are_related(name1, name2)
                or self.names_are_related_after_title_removal(name1, name2)
                or (
                    not corefs is None and self.names_are_in_coref(name1, name2, corefs)
                )
            ):
                G.add_edge(name1, name2)

        # delete the shortest path between two nodes if two names are found to be impossible to
        # to be a mention of the same character
        for (name1, name2) in combinations(G.nodes(), 2):
            if corefs is None:
                break
            # check if names dont have the same infered gender
            gender1 = self.infer_name_gender(name1, corefs)
            gender2 = self.infer_name_gender(name2, corefs)
            if gender1 != gender2 and not any(
                [g == Gender.UNKNOWN for g in (gender1, gender2)]
            ):
                for path in nx.all_shortest_paths(G, source=name1, target=name2):
                    G.remove_edges_from(path)
            # check if characters have the same last name but a
            # different first name
            human_name1 = HumanName(name1)
            human_name2 = HumanName(name2)
            if (
                human_name1.last == human_name2.last
                and human_name1.first != human_name2.first
            ):
                for path in nx.all_shortest_paths(G, source=name1, target=name2):
                    G.remove_edges_from(path)

        # create characters from the computed graph
        characters = [
            Character(frozenset(names)) for names in nx.connected_components(G)
        ]

        # filter characters based on the number of time they appear
        characters_c = Counter()
        for character in characters:
            for occurence in occurences:
                if occurence in character.names:
                    characters_c[character] += 1
        characters = [c for c in characters if characters_c[c] >= self.min_appearances]

        return {"characters": characters}

    def names_are_related_after_title_removal(self, name1: str, name2: str) -> bool:
        config.CONSTANTS.string_format = "{first} {middle} {last}"
        raw_name1 = HumanName(name1).full_name
        raw_name2 = HumanName(name2).full_name

        return raw_name1 == raw_name2 or self.hypocorism_gazetteer.are_related(
            raw_name1, raw_name2
        )

    def names_are_in_coref(
        self, name1: str, name2: str, corefs: List[CoreferenceChain]
    ):
        for coref_chain in corefs:
            if any([name1 == m.mention for m in coref_chain.mentions]) and any(
                [name2 == m.mention for m in coref_chain.mentions]
            ):
                return True
        return False

    def infer_name_gender(self, name: str, corefs: List[CoreferenceChain]) -> Gender:
        """Try to infer a name's gender"""
        # 1. try to infer gender based on honorifics
        #    TODO: add more gendered honorifics to renard.resources
        title = HumanName(name).title
        if any(
            [
                re.match(pattern, title)
                for pattern in (r"[Mm]r\.?", r"[Mm]\.?", r"[Ss]ir", r"[Ll]ord")
            ]
        ):
            return Gender.MALE
        elif title in any(
            [
                re.match(pattern, title)
                for pattern in (r"[Mm]iss", r"[Mm]r?s\.?", r"[Ll]ady")
            ]
        ):
            return Gender.FEMALE

        # 2. if 1. didn't succeed, inspect coreferences chain
        #    to see if if the name was coreferent with a
        #    gendered pronoun
        female_count = 0
        male_count = 0

        for coref_chain in corefs:
            mentions = {m.mention for m in coref_chain.mentions}
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
