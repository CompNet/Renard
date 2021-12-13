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

        if self.extract_dynamic_graph:
            dynamic_characters_graph = []

        for person, tokenidx in person_tokenidx:
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
                    if G.has_edge(person, close_character):
                        G.edges[person, close_character]["weight"] += 1
                    else:
                        G.add_edge(person, close_character, weight=1)
                if self.extract_dynamic_graph:
                    dynamic_characters_graph.append(copy.deepcopy(G))  # type: ignore

        if self.extract_dynamic_graph:
            return {
                "dynamic_characters_graph": dynamic_characters_graph,  # type: ignore
                "characters_graph": dynamic_characters_graph[-1]  # type: ignore
                if len(dynamic_characters_graph) > 0  # type: ignore
                else None,
            }
        return {"characters_graph": G}

    def needs(self) -> Set[str]:
        return {"tokens", "bio_tags", "characters"}

    def produces(self) -> Set[str]:
        production = {"characters_graph"}
        if self.extract_dynamic_graph:
            production.add("dynamic_characters_graph")
        return production
