import string, os
import pytest
from pytest import fixture
from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st
from more_itertools.recipes import flatten
from renard.pipeline.tokenization import NLTKTokenizer, BertTokenizer
from renard.pipeline.progress import get_progress_reporter


@fixture
def nltk_tokenizer() -> NLTKTokenizer:
    tokenizer = NLTKTokenizer()
    tokenizer._pipeline_init_("eng", get_progress_reporter(None))
    return tokenizer


# we suppress the `function_scoped_fixture` health check since we want
# to execute the `bert_tokenizer` fixture only once.
@given(input_text=st.text())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_nltk_tokens_and_sentences_are_aligned(
    input_text: str, nltk_tokenizer: NLTKTokenizer
):
    out_dict = nltk_tokenizer(input_text)
    assert out_dict["tokens"] == list(flatten(out_dict["sentences"]))


@fixture
def bert_tokenizer() -> BertTokenizer:
    tokenizer = BertTokenizer()
    tokenizer._pipeline_init_("eng", get_progress_reporter(None))
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
