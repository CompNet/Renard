from typing import List
import string, os
import pytest
from hypothesis import given
from hypothesis.control import assume
from hypothesis.strategies import lists, sampled_from
from transformers import BertTokenizerFast
from renard.ner_utils import NERDataset
from renard.pipeline.ner import ner_entities, score_ner, NERSamenounContextRetriever


@given(lists(sampled_from(("B-PER", "I-PER", "O")), min_size=1))
def test_score_same_tags(tags: List[str]):
    assume("B-PER" in tags)
    assert (1.0, 1.0, 1.0) == score_ner(tags, tags)


@given(lists(sampled_from(string.ascii_uppercase)))
def test_has_correct_number_of_entities(tokens: List[str]):
    bio_tags = ["B-PER" for _ in tokens]
    entities = ner_entities(tokens, bio_tags)
    assert len(entities) == len(tokens)


@pytest.mark.skipif(os.getenv("RENARD_TEST_ALL") != "1", reason="performance")
def test_retrieves_context():
    context_retriever = NERSamenounContextRetriever(1)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    sentences = [
        "this is some test sentence .".split(" "),
        "this is another test sentence .".split(" "),
    ]
    dataset = NERDataset(sentences, tokenizer)
    ctx_dataset = context_retriever(dataset)
    assert ctx_dataset.elements[0] == sentences[0] + ["[SEP]"] + sentences[1]
    assert ctx_dataset.elements[1] == sentences[1] + ["[SEP]"] + sentences[0]
    assert len(ctx_dataset.elements) == len(sentences)
    assert len(ctx_dataset._context_mask) == len(sentences)
