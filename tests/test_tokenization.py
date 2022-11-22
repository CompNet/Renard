import unittest, string, os
from hypothesis import given
import hypothesis.strategies as st
from more_itertools.recipes import flatten
from renard.pipeline.tokenization import NLTKTokenizer, BertTokenizer


class TestNLTKTokenizer(unittest.TestCase):
    """"""

    def setUp(self) -> None:
        self.tokenizer = NLTKTokenizer()

    @given(input_text=st.text())
    def test_tokens_and_sentences_are_aligned(self, input_text: str):
        out_dict = self.tokenizer(input_text)
        self.assertEqual(out_dict["tokens"], list(flatten(out_dict["sentences"])))
        self.assertEqual(out_dict["tokens"], list(flatten(out_dict["chapter_tokens"])))


@unittest.skipIf(os.getenv("RENARD_TEST_ALL") != "1", "skipped for performance")
class TestBertTokenizer(unittest.TestCase):
    """"""

    def setUp(self) -> None:
        self.tokenizer = BertTokenizer()

    @given(input_text=st.text(alphabet=string.ascii_letters, min_size=1))
    def test_tokens_and_sentences_are_aligned(self, input_text: str):
        out_dict = self.tokenizer(input_text)
        self.assertEqual(out_dict["tokens"], list(flatten(out_dict["sentences"])))
        self.assertEqual(out_dict["tokens"], list(flatten(out_dict["chapter_tokens"])))
