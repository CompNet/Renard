import string, os
import pytest
from pytest import fixture
from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st
from more_itertools.recipes import flatten
from renard.pipeline.tokenization import NLTKTokenizer, BertTokenizer
from renard.pipeline.progress import get_progress_reporter


@given(input_text=st.text())
def test_nltk_tokens_and_sentences_are_aligned(input_text: str):
    tokenizer = NLTKTokenizer()
    tokenizer._pipeline_init("eng", get_progress_reporter(None))
    out_dict = tokenizer(input_text)
    assert out_dict["tokens"] == list(flatten(out_dict["sentences"]))
    assert out_dict["tokens"] == list(flatten(out_dict["chapter_tokens"]))


@fixture
def bert_tokenizer() -> BertTokenizer:
    tokenizer = BertTokenizer()
    tokenizer._pipeline_init("eng", get_progress_reporter(None))
    return tokenizer


# we suppress the `function_scoped_fixture` health check since we want
# to execute the `bert_tokenizer` fixture only once.
@pytest.mark.skipif(os.getenv("RENARD_TEST_ALL") != "1", reason="performance")
@settings(deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(input_text=st.text(alphabet=string.ascii_letters, min_size=1))
def test_bert_tokens_and_sentences_are_aligned(
    input_text: str, bert_tokenizer: BertTokenizer
):
    out_dict = bert_tokenizer.__call__(input_text)
    assert out_dict["tokens"] == list(flatten(out_dict["sentences"]))
    assert out_dict["tokens"] == list(flatten(out_dict["chapter_tokens"]))
