import string, os
import pytest
from pytest import fixture
from hypothesis import given, settings, HealthCheck
import hypothesis.strategies as st
from more_itertools.recipes import flatten
from renard.pipeline.tokenization import NLTKTokenizer, StanzaTokenizer
from renard.pipeline.progress import get_progress_reporter


@fixture
def nltk_tokenizer() -> NLTKTokenizer:
    tokenizer = NLTKTokenizer()
    tokenizer._pipeline_init_("eng", progress_reporter=get_progress_reporter(None))
    return tokenizer


# we suppress the `function_scoped_fixture` health check since we want
# to execute the `nltk_tokenizer` fixture only once.
@given(text=st.text())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_nltk_tokens_and_sentences_are_aligned(
    text: str, nltk_tokenizer: NLTKTokenizer
):
    out_dict = nltk_tokenizer(text)
    assert out_dict["tokens"] == list(flatten(out_dict["sentences"]))


@given(text=st.text())
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_nltk_len_tokens_leq_len_char2token(text: str, nltk_tokenizer: NLTKTokenizer):
    out_dict = nltk_tokenizer(text)
    assert len(out_dict["tokens"]) <= len(out_dict["char2token"])


@fixture
def stanza_tokenizer() -> StanzaTokenizer:
    tokenizer = StanzaTokenizer()
    tokenizer._pipeline_init_("eng", progress_reporter=get_progress_reporter(None))
    return tokenizer


@pytest.mark.skipif(
    os.getenv("RENARD_TEST_OPTDEP_STANZA") != "1", reason="optional stanza dependency"
)
@pytest.mark.skipif(os.getenv("RENARD_TEST_SLOW") != "1", reason="performance")
@given(text=st.text(max_size=32))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
def test_stanza_tokens_and_sentences_are_aligned(
    text: str, stanza_tokenizer: StanzaTokenizer
):
    out_dict = stanza_tokenizer(text)
    assert out_dict["tokens"] == list(flatten(out_dict["sentences"]))


@pytest.mark.skipif(
    os.getenv("RENARD_TEST_OPTDEP_STANZA") != "1", reason="optional stanza dependency"
)
@pytest.mark.skipif(os.getenv("RENARD_TEST_SLOW") != "1", reason="performance")
@given(text=st.text(max_size=32))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=None)
def test_stanza_len_tokens_leq_len_char2token(
    text: str, stanza_tokenizer: StanzaTokenizer
):
    out_dict = stanza_tokenizer(text)
    assert len(out_dict["tokens"]) <= len(out_dict["char2token"])
