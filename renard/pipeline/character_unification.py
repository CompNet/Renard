from typing import Any, Dict, List, FrozenSet, Set, Optional, Tuple, Union, Literal
import re, sys
from itertools import combinations
from collections import defaultdict, Counter
from dataclasses import dataclass
from nameparser import HumanName
from nameparser.config import Constants
from renard.gender import Gender
from renard.pipeline.core import Mention, PipelineStep
from renard.pipeline.ner import NEREntity
from renard.pipeline.progress import ProgressReporter
from renard.resources.hypocorisms import HypocorismGazetteer
from renard.resources.pronouns import is_a_female_pronoun, is_a_male_pronoun
from renard.resources.determiners import singular_determiners
from renard.resources.titles import is_a_male_title, is_a_female_title, all_titles


@dataclass(eq=True, frozen=True)
class Character:
    names: FrozenSet[str]
    mentions: List[Mention]
    gender: Gender = Gender.UNKNOWN

    def longest_name(self) -> str:
        return max(self.names, key=len)

    def shortest_name(self) -> str:
        return min(self.names, key=len)

    def most_frequent_name(self) -> Optional[str]:
        c = Counter([" ".join(mention.tokens) for mention in self.mentions])
        c = {c: count for c, count in c.items() if c in self.names}
        if len(c) == 0:
            return self.longest_name()
        return max(c, key=c.get)  # type: ignore

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.names)))

    def __repr__(self) -> str:
        return f"<{self.most_frequent_name()}, {self.gender}, {len(self.mentions)} mentions>"


def _assign_coreference_mentions(
    characters: List[Character], corefs: List[List[Mention]]
) -> List[Character]:
    """Assign mentions to characters from coreference chains.

    Each coreference chain is assigned to the character whose names
    have the most occurences in the chain.  When it seems that no
    characters appear in the chain, it is discarded.

    :param characters: A list of characters, where ``character.names``
        contains the list of all names of a character.
    :param corefs:
    """

    char_mentions: Dict[Character, Set[Mention]] = {
        character: set(character.mentions) for character in characters
    }

    # we assign each chain to the character with highest name
    # occurence in it
    for chain in corefs:
        if len(char_mentions) == 0:
            break
        # determine the characters with the highest number of
        # occurences
        occ_counter = {}
        for character in char_mentions.keys():
            occ_counter[character] = sum(
                [
                    1 if " ".join(mention.tokens) in character.names else 0
                    for mention in chain
                ]
            )
        best_character = max(occ_counter, key=occ_counter.get)  # type: ignore

        # no character occurences in this chain: don't assign
        # it to any character
        if occ_counter[best_character] == 0:
            continue

        # assign the chain to the character with the most occurences
        for mention in chain:
            if not mention in char_mentions[best_character]:
                char_mentions[best_character].add(mention)

    return [
        Character(c.names, sorted(mentions, key=lambda m: m.start_idx), c.gender)
        for c, mentions in char_mentions.items()
    ]


class NaiveCharacterUnifier(PipelineStep):
    """A basic character unifier using NER"""

    def __init__(self, min_appearances: int = 0) -> None:
        """
        :param min_appearances: minimum number of appearances of a
            character for it to be valid
        """
        self.min_appearances = min_appearances
        # a default value, will be est by _pipeline_init_
        self.character_ner_tag = "PER"
        super().__init__()

    def _pipeline_init_(self, lang: str, character_ner_tag: str, **kwargs):
        self.character_ner_tag = character_ner_tag

    def __call__(
        self,
        text: str,
        entities: List[NEREntity],
        corefs: Optional[List[List[Mention]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        :param text:
        :param tokens:
        :param entities:
        """
        persons = [e for e in entities if e.tag == self.character_ner_tag]

        characters = defaultdict(list)
        for entity in persons:
            characters[" ".join(entity.tokens)].append(entity)

        characters = [
            Character(frozenset([name]), mentions)
            for name, mentions in characters.items()
        ]

        if not corefs is None:
            characters = _assign_coreference_mentions(characters, corefs)

        # filter characters based on the number of time they appear
        characters = [c for c in characters if len(c.mentions) >= self.min_appearances]

        return {"characters": characters}

    def supported_langs(self) -> Union[Set[str], Literal["any"]]:
        return "any"

    def needs(self) -> Set[str]:
        return {"entities"}

    def optional_needs(self) -> Set[str]:
        return {"corefs"}

    def production(self) -> Set[str]:
        return {"characters"}


class GraphRulesCharacterUnifier(PipelineStep):
    """Unify characters by creating a graph where mentions are
    linked when they refer to the same character, and then
    merging this graph nodes.

    .. note::

        This algorithm is inspired from Vala et al., 2015.
    """

    def __init__(
        self,
        min_appearances: int = 0,
        additional_hypocorisms: Optional[List[Tuple[str, List[str]]]] = None,
        link_corefs_mentions: bool = False,
        ignore_lone_titles: Optional[Set[str]] = None,
        ignore_leading_determiner: bool = False,
    ) -> None:
        """
        :param min_appearances: minimum number of appearances of a
            character for it to be considered valid.
        :param additional_hypocorisms: a tuple of additional
            hypocorisms.  Each hypocorism is a tuple where the first
            element is a name, and the second element is a set of
            nicknames associated with it
        :param link_corefs_mentions: if ``True``, will also use
            coreference resolution to link names between them.  This
            is disabled by default since a coreference model can
            extract a lot of spurious links.  However, linking by
            coref is sometimes the only way to resolve a character
            alias.
        :param ignore_lone_titles: a set of titles to ignore when they
            stand on their own.  This avoids extracting false
            positives characters such as 'Mr.' or 'Miss'.
        :param ignore_leading_determiner: if ``True``, will ignore the
            leading determiner when applying unification rules.  This
            is useful if the NER model used in the pipeline adds
            leading determiners as part of entites.
        """
        self.min_appearances = min_appearances
        self.additional_hypocorisms = additional_hypocorisms
        self.link_corefs_mentions = link_corefs_mentions
        self.ignore_lone_titles = ignore_lone_titles or set()
        self.character_ner_tag = "PER"  # a default value, will be set by _pipeline_init
        self.ignore_leading_determiner = ignore_leading_determiner

        super().__init__()

    def _pipeline_init_(self, lang: str, character_ner_tag: str, **kwargs):
        self.hypocorism_gazetteer = HypocorismGazetteer(lang=lang)
        if not self.additional_hypocorisms is None:
            for name, nicknames in self.additional_hypocorisms:
                self.hypocorism_gazetteer._add_hypocorism_(name, nicknames)

        self.character_ner_tag = character_ner_tag

        return super()._pipeline_init_(lang, **kwargs)

    def __call__(
        self,
        entities: List[NEREntity],
        corefs: Optional[List[List[Mention]]] = None,
        **kwargs: dict,
    ) -> Dict[str, Any]:
        import networkx as nx

        mentions = [m for m in entities if m.tag == self.character_ner_tag]
        mentions_str = set(
            filter(
                lambda m: not m in self.ignore_lone_titles,
                map(lambda m: " ".join(m.tokens), mentions),
            )
        )

        # * create a graph where each node is a mention detected by NER
        G = nx.Graph()
        for mention_str in mentions_str:
            G.add_node(mention_str)

        # * HumanName local configuration - dependant on language
        hname_constants = self._make_hname_constants()

        # * link nodes based on several rules
        for name1, name2 in combinations(G.nodes(), 2):

            # preprocess name when needed
            pname1 = self._preprocess_name(name1)
            pname2 = self._preprocess_name(name2)

            # is one name a known hypocorism of the other ? (also
            # checks if both names are the same)
            if self.hypocorism_gazetteer.are_related(pname1, pname2):
                G.add_edge(name1, name2)
                continue

            # if we remove the title, is one name related to the other
            # ?
            if self.names_are_related_after_title_removal(
                pname1, pname2, hname_constants
            ):
                G.add_edge(name1, name2)
                continue

            # add an edge if two characters have the same family names
            human_name1 = HumanName(pname1, constants=hname_constants)
            human_name2 = HumanName(pname2, constants=hname_constants)
            if (
                len(human_name1.last) > 0
                and human_name1.last.lower() == human_name2.last.lower()
            ):
                G.add_edge(name1, name2)
                continue
            # add an edge if two characters have the same first names
            if (
                len(human_name1.first) > 0
                and human_name1.first.lower() == human_name2.first.lower()
            ):
                G.add_edge(name1, name2)
                continue

            # if coreferences are available, check if both names are
            # in a coref chain
            if not corefs is None and self.link_corefs_mentions:
                if self.names_are_in_coref(name1, name2, corefs):
                    G.add_edge(name1, name2)

        # we assign a gender to each name when corefs are available
        for name in G.nodes():
            G.nodes[name]["gender"] = self.infer_name_gender(
                name, corefs, hname_constants
            )

        # * delete the shortest path between two nodes if two names
        #   are found to be impossible to be a mention of the same
        #   character
        def try_remove_edges(edges):
            try:
                G.remove_edges_from(edges)
            except nx.NetworkXNoPath:
                pass

        for name1, name2 in combinations(G.nodes(), 2):

            # preprocess names when needed
            pname1 = self._preprocess_name(name1)
            pname2 = self._preprocess_name(name2)

            # check if characters have the same last name but a
            # different first name.
            human_name1 = HumanName(pname1, constants=hname_constants)
            human_name2 = HumanName(pname2, constants=hname_constants)
            if (
                len(human_name1.last) > 0
                and len(human_name2.last) > 0
                and len(human_name1.first) > 0
                and len(human_name2.first) > 0
                and human_name1.last == human_name2.last
                and human_name1.first != human_name2.first
            ):
                try_remove_edges(nx.all_shortest_paths(G, source=name1, target=name2))
                continue

            # check if names dont have the same infered gender
            gender1 = G.nodes[name1]["gender"]
            gender2 = G.nodes[name2]["gender"]
            if (
                gender1 != gender2
                and not gender1 == Gender.UNKNOWN
                and not gender2 == Gender.UNKNOWN
            ):
                try_remove_edges(nx.all_shortest_paths(G, source=name1, target=name2))

        # create characters from the computed graph
        characters = []
        for names in nx.connected_components(G):
            # choose gender using a majority vote
            genders = [G.nodes[name].get("gender", Gender.UNKNOWN) for name in names]
            counter = Counter(genders)
            gender = max(counter, key=counter.get)  # type: ignore
            characters.append(
                Character(
                    frozenset(names),
                    [m for m in mentions if " ".join(m.tokens) in names],
                    gender,
                )
            )

        # link characters to all of to their coreferential mentions
        # (pronouns...)
        if not corefs is None:
            characters = _assign_coreference_mentions(characters, corefs)

        # filter characters based on the number of time they appear
        characters = [
            c
            for c in characters
            if len([m for m in c.mentions if " ".join(m.tokens) in c.names])
            >= self.min_appearances
        ]

        return {"characters": characters}

    def _preprocess_name(self, name) -> str:
        if self.ignore_leading_determiner:
            if not self.lang in singular_determiners:
                print(
                    f"[warning] can't ignore leading determiners for {self.lang}",
                    file=sys.stderr,
                )
            for determiner in singular_determiners.get(self.lang, []):
                name = re.sub(f"^{determiner} ", " ", name, flags=re.I)
        return name

    def _make_hname_constants(self) -> Constants:
        if self.lang == "eng":
            return Constants()
        if self.lang == "fra":
            hname_constants = Constants()
            for title in all_titles["fra"]:
                hname_constants.titles.add(title)
            return hname_constants
        raise ValueError(f"unsupported language: {self.lang}")

    def names_are_related_after_title_removal(
        self, name1: str, name2: str, hname_constants: Constants
    ) -> bool:
        """Check if two names are related after removing their titles"""
        old_string_format = hname_constants.string_format
        hname_constants.string_format = "{first} {middle} {last}"
        raw_name1 = HumanName(name1, constants=hname_constants).full_name
        raw_name2 = HumanName(name2, constants=hname_constants).full_name
        hname_constants.string_format = old_string_format

        if raw_name1 == "" or raw_name2 == "":
            return False

        return (
            raw_name1.lower() == raw_name2.lower()
            or self.hypocorism_gazetteer.are_related(raw_name1, raw_name2)
        )

    def names_are_in_coref(
        self, name1: str, name2: str, corefs: List[List[Mention]]
    ) -> bool:
        once_together = False
        for coref_chain in corefs:
            name1_in = any([name1 == " ".join(m.tokens) for m in coref_chain])
            name2_in = any([name2 == " ".join(m.tokens) for m in coref_chain])
            if name1_in == (not name2_in):
                return False
            elif name1_in and name2_in:
                once_together = True
        return once_together

    def infer_name_gender(
        self,
        name: str,
        corefs: Optional[List[List[Mention]]],
        hname_constants: Constants,
    ) -> Gender:
        """Try to infer a name's gender

        :param name:
        :param corefs:
        :param hname_constants: HumanName constants
        """
        # 1. try to infer gender based on honorifics
        title = HumanName(name, constants=hname_constants).title
        if title != "":
            if is_a_male_title(title, lang=self.lang):
                return Gender.MALE
            elif is_a_female_title(title, lang=self.lang):
                return Gender.FEMALE

        # 2. if 1. didn't succeed, inspect coreferences chain
        #    to see if if the name was coreferent with a
        #    gendered pronoun
        if corefs is None:
            return Gender.UNKNOWN

        female_count = 0
        male_count = 0

        for coref_chain in corefs:
            mentions = {" ".join(m.tokens) for m in coref_chain}
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
        return {"entities"}

    def optional_needs(self) -> Set[str]:
        return {"corefs"}

    def supported_langs(self) -> Union[Set[str], Literal["any"]]:
        return {"eng", "fra"}

    def production(self) -> Set[str]:
        return {"characters"}
