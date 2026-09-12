from typing import List, Type
import string, os
import pytest
from hypothesis import given
from hypothesis.control import assume
from hypothesis.strategies import lists, sampled_from
from transformers import BertTokenizerFast
from renard.pipeline.progress import get_progress_reporter
from renard.ner_utils import NERDataset
from renard.pipeline.ner import ner_entities, score_ner, BertNamedEntityRecognizer
from renard.pipeline.ner.retrieval import (
    NERBM25ContextRetriever,
    NERContextRetriever,
    NEREnsembleContextRetriever,
    NERNeighborsContextRetriever,
    NERSamenounContextRetriever,
    NERNeuralContextRetriever,
)


@pytest.mark.skipif(
    os.getenv("RENARD_TEST_OPTDEP_SEQEVAL") != "1",
    reason="not testing seqeval based functions",
)
@given(lists(sampled_from(("B-PER", "I-PER", "O")), min_size=1))
def test_score_same_tags(tags: List[str]):
    assume("B-PER" in tags)
    assert (1.0, 1.0, 1.0) == score_ner(tags, tags)


@given(lists(sampled_from(string.ascii_uppercase)))
def test_has_correct_number_of_entities(tokens: List[str]):
    bio_tags = ["B-PER" for _ in tokens]
    entities = ner_entities(tokens, bio_tags)
    assert len(entities) == len(tokens)


@pytest.mark.skipif(os.getenv("RENARD_TEST_SLOW") != "1", reason="performance")
def test_run_with_context_retriever():
    ner_step = BertNamedEntityRecognizer(
        context_retriever=NERNeighborsContextRetriever(k=2)
    )
    ner_step._pipeline_init_(lang="eng", progress_reporter=get_progress_reporter(None))
    # known crash in Renard==0.7.1
    sentences = [
        "Whether i shall turn out to be the hero of my own life , or whether that station will be held by anybody else , these pages must show .".split(),
        "To begin my life with the beginning of my life , i record that i was born ( as i have been informed and believe ) on a friday , at twelve o'clock at night .".split(),
        "This was the fault of Dr. Strange .".split(),
    ]
    tokens = [token for tokens in sentences for token in tokens]
    _ = ner_step(tokens, sentences)


@pytest.mark.skipif(os.getenv("RENARD_TEST_SLOW") != "1", reason="performance")
@pytest.mark.parametrize(
    "retriever_class", [NERSamenounContextRetriever, NERBM25ContextRetriever]
)
def test_retrieves_context(retriever_class: Type[NERContextRetriever]):
    context_retriever = retriever_class(1)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    sentences = [
        "this is some test sentence .".split(" "),
        "this is another test sentence .".split(" "),
    ]
    dataset = NERDataset(sentences, tokenizer)
    ctx_dataset = context_retriever(dataset)
    assert ctx_dataset.elements[0] == sentences[0] + sentences[1]
    assert ctx_dataset.elements[1] == sentences[0] + sentences[1]
    assert len(ctx_dataset.elements) == len(sentences)
    assert len(ctx_dataset._context_mask) == len(sentences)


@pytest.mark.skipif(os.getenv("RENARD_TEST_SLOW") != "1", reason="performance")
def test_neural_retrieves_context():
    context_retriever = NERNeuralContextRetriever(
        NEREnsembleContextRetriever(
            [
                NERSamenounContextRetriever(1),
                NERBM25ContextRetriever(1),
                NERNeighborsContextRetriever(2),
            ],
            k=4,
        ),
        k=1,
    )
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    sentences = [
        "this is some test sentence .".split(" "),
        "this is another test sentence .".split(" "),
    ]
    dataset = NERDataset(sentences, tokenizer)
    ctx_dataset = context_retriever(dataset)
    assert ctx_dataset.elements[0] == sentences[0] + sentences[1]
    assert ctx_dataset.elements[1] == sentences[0] + sentences[1]
    assert len(ctx_dataset.elements) == len(sentences)
    assert len(ctx_dataset._context_mask) == len(sentences)
