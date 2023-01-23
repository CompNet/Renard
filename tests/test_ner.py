from typing import List
import string
from hypothesis import given
from hypothesis.control import assume
from hypothesis.strategies import lists, sampled_from
from renard.pipeline.ner import ner_entities, score_ner


@given(lists(sampled_from(("B-PER", "I-PER", "O")), min_size=1))
def test_score_same_tags(tags: List[str]):
    assume("B-PER" in tags)
    assert (1.0, 1.0, 1.0) == score_ner(tags, tags)


@given(lists(sampled_from(string.ascii_uppercase)))
def test_has_correct_number_of_entities(tokens: List[str]):
    bio_tags = ["B-PER" for _ in tokens]
    entities = ner_entities(tokens, bio_tags)
    assert len(entities) == len(tokens)
