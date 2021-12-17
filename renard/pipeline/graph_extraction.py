from typing import Dict, Any, List, Set, Optional
import copy

from renard.pipeline.ner import bio_entities
from renard.pipeline.core import PipelineStep


class CoOccurencesGraphExtractor(PipelineStep):
    """A simple character graph extractor using co-occurences"""

    def __init__(
        self, co_occurences_dist: int, extract_dynamic_graph: bool = False
    ) -> None:
        self.co_occurences_dist = co_occurences_dist
        self.extract_dynamic_graph = extract_dynamic_graph

    def __call__(
        self,
        text: str,
        tokens: List[str],
        bio_tags: List[str],
        characters: Set[str],
        **kwargs
    ) -> Dict[str, Any]:
        """Extract a characters graph

        :param tokens:
        :param bio_tags:
        :param characters:

        :return: a ``dict`` with key ``'characters_graph'`` and a
            ``networkx.Graph`` as value.

        .. note::

            Although ``networkx.Graph`` doesnt really support dynamic
            graphs, when ``self.extract_dynamic_graph`` is set to
            ``True``, the returned graph has the nice property that
            exporting it to `gexf` format using ``G.write_gexf()``
            will produce a correct dynamic graph that can be read by
            Gephi. Because of a limitation in networkx, the dynamic
            weight attribute is stored as ``'dweight'`` instead of
            ``'weight'``.

        """
        assert len(tokens) == len(bio_tags)

        import networkx as nx

        G = nx.Graph()

        character_to_last_appearance: Dict[str, Optional[int]] = {
            character: None for character in characters
        }

        # (person, token index)
        person_tokenidx = [
            (e[0], e[2])
            for e in bio_entities(tokens, bio_tags)
            if e[1].startswith("PER")
        ]

        for i, (person, tokenidx) in enumerate(person_tokenidx):
            if person in characters:
                character_to_last_appearance[person] = tokenidx
                close_characters = [
                    c
                    for c, last_appearance in character_to_last_appearance.items()
                    if not last_appearance is None
                    and tokenidx - last_appearance <= self.co_occurences_dist
                    and not c == person
                ]
                for close_character in close_characters:
                    if not G.has_edge(person, close_character):
                        if self.extract_dynamic_graph:
                            G.add_edge(person, close_character)
                            # dynamic graphs : the edge exists starting timestep i
                            G.edges[person, close_character]["start"] = i
                            # dynamic graphs : the weight attribute is an empty series
                            G.edges[person, close_character]["dweight"] = []
                        else:
                            G.add_edge(person, close_character, weight=0)
                    if self.extract_dynamic_graph:
                        # dynamic graphs : add a new entry to the weight series
                        # according to networkx source code, each entry must be
                        # of the form [value, start, end]
                        weights = G.edges[person, close_character]["dweight"]
                        if len(weights) != 0:
                            # end of last weight attribute
                            weights[-1][-1] = i
                        # value, start and end of current weight attribute
                        last_weight_value = weights[-1][0] if len(weights) > 0 else 0
                        G.edges[person, close_character]["dweight"].append(
                            [float(last_weight_value) + 1, i, len(person_tokenidx)]
                        )
                    else:
                        G.edges[person, close_character]["weight"] += 1

        return {"characters_graph": G}

    def needs(self) -> Set[str]:
        return {"tokens", "bio_tags", "characters"}

    def produces(self) -> Set[str]:
        return {"characters_graph"}
