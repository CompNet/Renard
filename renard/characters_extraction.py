from collections import defaultdict
from typing import Any, Dict, List, Set
from renard.pipeline import PipelineStep


class NaiveCharactersExtractor(PipelineStep):
    """A basic character extractor using NER

    needs the following pipeline attributes :

    - ``tokens``
    - ``bio_tags``

    :ivar min_appearance:
    """

    def __init__(self, min_appearance: int) -> None:
        self.min_appearance = min_appearance

    def __call__(
        self, text: str, tokens: List[str], bio_tags: List[str], **kwargs
    ) -> Dict[str, Any]:

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
            entity for entity, count in entities.items() if count > self.min_appearance
        ]

        return {"characters": characters}

    def needs(self) -> Set[str]:
        return {"tokens", "bio_tags"}

    def produces(self) -> Set[str]:
        return {"characters"}
