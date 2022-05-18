from renard.pipeline.characters_extraction import Character
from typing import Dict, Any, List, Set, Optional, Tuple, cast, Literal

import networkx as nx
import numpy as np
from more_itertools import windowed

from renard.pipeline.ner import ner_entities
from renard.pipeline.core import PipelineStep


class CoOccurencesGraphExtractor(PipelineStep):
    """A simple character graph extractor using co-occurences"""

    def __init__(
        self,
        co_occurences_dist: int,
        dynamic: Optional[Literal["nx", "gephi"]] = None,
        dynamic_window: Optional[int] = None,
        dynamic_overlap: int = 0,
    ) -> None:
        """
        :param co_occurences_dist: max accepted distance between two
            character appearances.

        :param dynamic: either ``None``, or one of ``{'nx', 'gephi'}``

                - if ``None`` (the defaul), a ``nx.graph`` is
                  extracted

                - if ``'nx'``, several ``nx.graph`` are extracted.  In
                  that case, ``dynamic_window`` *must* be specified,
                  and overlap *can* be specified.

                - if ``'gephi'``, a single ``nx.graph`` is extracted.
                  This graph has the nice property that exporting it
                  to `gexf` format using ``G.write_gexf()`` will
                  produce a correct dynamic graph that can be read by
                  Gephi.  Because of a limitation in networkx, the
                  dynamic weight attribute is stored as ``dweight``
                  instead of ``weight``.

        :param dynamic_window: dynamic window, in number of
            interactions.  a dynamic window of `n` means that each
            returned graph will be formed by `n` interactions.

        :param dynamic_overlap: overlap, in number of interactions.
        """

        self.co_occurences_dist = co_occurences_dist

        if not dynamic is None:
            assert dynamic in {"nx", "gephi"}
            if dynamic == "nx":
                assert not dynamic_window is None and dynamic_window > 0
                assert dynamic_window > dynamic_overlap
        self.dynamic = dynamic
        self.dynamic_window = dynamic_window
        self.dynamic_overlap = dynamic_overlap
        super().__init__()

    def __call__(
        self,
        text: str,
        tokens: List[str],
        bio_tags: List[str],
        characters: Set[Character],
        **kwargs
    ) -> Dict[str, Any]:
        """Extract a characters graph

        :param tokens:
        :param bio_tags:
        :param characters:

        :return: a ``dict`` with key ``'characters_graph'`` and a
            ``networkx.Graph`` or a list of ``networkx.Graph`` as
            value.
        """
        assert len(tokens) == len(bio_tags)

        # greedily assign mentions
        character_tokenidx = []
        for entity in ner_entities(tokens, bio_tags):
            if not entity.tag.startswith("PER"):
                continue
            mention = " ".join(entity.tokens)
            for character in characters:
                if mention in character.names:
                    character_tokenidx.append((character, entity.start_idx))
                    break

        if self.dynamic == "gephi":
            return {
                "characters_graph": self._extract_gephi_dynamic_graph(
                    character_tokenidx
                )
            }
        elif self.dynamic == "nx":
            # we already checked this at __init__ time
            self.dynamic_window = cast(int, self.dynamic_window)
            return {
                "characters_graph": self._extract_dynamic_graph(
                    character_tokenidx, self.dynamic_window, self.dynamic_overlap
                )
            }
        return {"characters_graph": self._extract_graph(character_tokenidx)}

    def _extract_graph(self, character_tokenidx: List[Tuple[Character, int]]):

        # co-occurence matrix, where C[i][j] is 1 when appearance
        # i co-occur with j if i < j, or 0 when it doesn't
        C = np.zeros((len(character_tokenidx), len(character_tokenidx)))
        for i, (char1, token_idx) in enumerate(character_tokenidx):
            # check ahead for co-occurences
            for j, (char2, token_idx2) in enumerate(character_tokenidx[i + 1 :]):
                if token_idx2 - token_idx > self.co_occurences_dist:
                    # dist between current token and future token is
                    # too great : we finished co-occurences search for
                    # the current token
                    break
                # ignore co-occurences with self
                if char1 == char2:
                    continue
                # record co_occurence
                C[i][i + 1 + j] = 1

        # construct graph from co-occurence matrix
        G = nx.Graph()
        for i in range(len(character_tokenidx)):
            for j in range(len(character_tokenidx)):
                if C[i][j] == 0:
                    continue
                char1 = character_tokenidx[i][0]
                char2 = character_tokenidx[j][0]
                if not G.has_edge(char1, char2):
                    G.add_edge(char1, char2, weight=0)
                G.edges[char1, char2]["weight"] += 1

        return G

    def _extract_dynamic_graph(
        self, character_tokenidx: List[Tuple[Character, int]], window: int, overlap: int
    ) -> List[nx.Graph]:
        return [
            self._extract_graph([elt for elt in ct if not elt is None])
            for ct in windowed(character_tokenidx, window, step=window - overlap)
        ]

    def _extract_gephi_dynamic_graph(
        self, character_tokenidx: List[Tuple[Character, int]]
    ) -> nx.Graph:
        # keep only longest name in graph node : possible only if it is unique
        # TODO: might want to try and get shorter names if longest names aren't
        #       unique
        characters = set([e[0] for e in character_tokenidx])
        character_names = set([e[0].longest_name() for e in character_tokenidx])
        assert len(character_names) == len(characters)

        G = nx.Graph()

        name_tokenidx = [(e[0].longest_name(), e[1]) for e in character_tokenidx]
        character_to_last_appearance: Dict[str, Optional[int]] = {
            name: None for name in character_names
        }

        for i, (character, tokenidx) in enumerate(name_tokenidx):
            if not character in characters:
                continue
            character_to_last_appearance[character] = tokenidx
            close_characters = [
                c
                for c, last_appearance in character_to_last_appearance.items()
                if not last_appearance is None
                and tokenidx - last_appearance <= self.co_occurences_dist
                and not c == character
            ]
            for close_character in close_characters:
                if not G.has_edge(character, close_character):
                    G.add_edge(character, close_character)
                    G.edges[character, close_character]["start"] = i
                    G.edges[character, close_character]["dweight"] = []
                # add a new entry to the weight series according to networkx
                # source code, each entry must be of the form
                # [value, start, end]
                weights = G.edges[character, close_character]["dweight"]
                if len(weights) != 0:
                    # end of last weight attribute
                    weights[-1][-1] = i
                # value, start and end of current weight attribute
                last_weight_value = weights[-1][0] if len(weights) > 0 else 0
                G.edges[character, close_character]["dweight"].append(
                    [float(last_weight_value) + 1, i, len(name_tokenidx)]
                )

        return G

    def needs(self) -> Set[str]:
        return {"tokens", "bio_tags", "characters"}

    def production(self) -> Set[str]:
        return {"characters_graph"}
