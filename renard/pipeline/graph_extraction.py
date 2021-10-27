from typing import Dict, Any, List, Set, Optional
from renard.pipeline.ner import bio_entities
from renard.pipeline.core import PipelineStep


class CoOccurencesGraphExtractor(PipelineStep):
    def __init__(self, co_occurences_dist: int) -> None:
        self.co_occurences_dist = co_occurences_dist

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

        return {"characters_graph": G}

    def needs(self) -> Set[str]:
        return {"tokens", "bio_tags", "characters"}

    def produces(self) -> Set[str]:
        return {"characters_graph"}
