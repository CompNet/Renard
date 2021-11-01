from typing import List, Tuple
import unittest, random
from hypothesis import given
from hypothesis.control import assume
from hypothesis.strategies import lists, tuples, sampled_from, integers
from renard.pipeline.ner import score_ner


class TestScoreNer(unittest.TestCase):
    """"""

    def test_score_empty_tagset(self):
        self.assertEqual((None, None, None), score_ner([], []))

    @given(
        integers(min_value=1, max_value=100).flatmap(
            lambda n: tuples(
                lists(sampled_from(("O")), min_size=n, max_size=n),
                lists(sampled_from(("B-PER", "I-PER", "O")), min_size=n, max_size=n),
            )
        )
    )
    def test_score_zero_recall(self, tags: Tuple[List[str], List[str]]):
        pred_tags, ref_tags = tags
        pred_tags.append("O")
        ref_tags.append("B-PER")
        random.shuffle(ref_tags)
        self.assertEqual(0.0, score_ner(pred_tags, ref_tags)[1])

    @given(
        integers(min_value=1, max_value=100).flatmap(
            lambda n: tuples(
                lists(sampled_from(("B-PER", "I-PER", "O")), min_size=n, max_size=n),
                lists(sampled_from(("O")), min_size=n, max_size=n),
            )
        )
    )
    def test_score_zero_precision(self, tags: Tuple[List[str], List[str]]):
        pred_tags, ref_tags = tags
        pred_tags.append("B-PER")
        random.shuffle(pred_tags)
        ref_tags.append("O")
        self.assertEqual(0.0, score_ner(pred_tags, ref_tags)[0])

    @given(lists(sampled_from(("B-PER", "I-PER", "O")), min_size=1))
    def test_score_same_tags(self, tags: List[str]):
        assume("B-PER" in tags)
        self.assertEqual((1.0, 1.0, 1.0), score_ner(tags, tags))


if __name__ == "__main__":
    unittest.main()
