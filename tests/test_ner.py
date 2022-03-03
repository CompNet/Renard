from typing import List
import unittest, string
from hypothesis import given
from hypothesis.control import assume
from hypothesis.strategies import lists, sampled_from
from renard.pipeline.ner import bio_entities, score_ner


class TestScoreNer(unittest.TestCase):
    """"""

    @given(lists(sampled_from(("B-PER", "I-PER", "O")), min_size=1))
    def test_score_same_tags(self, tags: List[str]):
        assume("B-PER" in tags)
        self.assertEqual((1.0, 1.0, 1.0), score_ner(tags, tags))


class TestBioEntities(unittest.TestCase):
    """"""

    @given(lists(sampled_from(string.ascii_uppercase)))
    def test_has_correct_number_of_entities(self, tokens: List[str]):
        bio_tags = ["B-PER" for _ in tokens]
        entities = bio_entities(tokens, bio_tags)
        self.assertEqual(len(entities), len(tokens))


if __name__ == "__main__":
    unittest.main()
