from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional, Union, Dict, Tuple
import os, re
import itertools as it
import functools as ft
from more_itertools import flatten
import torch
from torch.utils.data import Dataset
from datasets import Dataset as HGDataset
from datasets import Sequence, ClassLabel
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    PreTrainedTokenizerFast,
    PreTrainedModel,
    Trainer,
    TrainingArguments,
)
from transformers.tokenization_utils_base import BatchEncoding

if TYPE_CHECKING:
    from renard.pipeline.ner import NEREntity


class DataCollatorForTokenClassificationWithBatchEncoding:
    """Same as ``transformers.DataCollatorForTokenClassification``,
    except it correctly returns a ``BatchEncoding`` object with
    correct ``encodings`` attribute.

    Don't know why this is not the default ?
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        pad_to_multiple_of: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_token_id = {"label": -100, "labels": -100}

    def __call__(self, features: List[dict]) -> Union[dict, BatchEncoding]:
        keys = features[0].keys()
        sequence_len = max([len(f["input_ids"]) for f in features])

        # We do the padding and collating manually instead of calling
        # self.tokenizer.pad, because pad does not work on arbitrary
        # features.
        batch = BatchEncoding({})
        for key in keys:
            if self.tokenizer.padding_side == "right":
                batch[key] = [
                    f[key]
                    + [self.pad_token_id.get(key, 0)] * (sequence_len - len(f[key]))
                    for f in features
                ]
            else:
                batch[key] = [
                    [
                        self.pad_token_id.get(key, 0) * (sequence_len - len(f[key]))
                        + f[key]
                        for f in features
                    ]
                ]

        batch._encodings = [f.encodings[0] for f in features]

        for k, v in batch.items():
            batch[k] = torch.tensor(v)

        return batch


class NERDataset(Dataset):
    """
    :ivar _context_mask: for each element, a mask indicating which
        tokens are part of the context (0 for context, 1 for text on
        which to perform inference).  The mask allows to discard
        predictions made for context at inference time, even though
        the context can still be passed as input to the model.
    """

    def __init__(
        self,
        elements: List[List[str]],
        tokenizer: PreTrainedTokenizerFast,
        context_mask: Optional[List[List[int]]] = None,
    ) -> None:
        self.elements = elements

        if context_mask:
            assert all(
                [len(cm) == len(elt) for elt, cm in zip(self.elements, context_mask)]
            )
        self._context_mask = context_mask or [[1] * len(elt) for elt in self.elements]

        self.tokenizer = tokenizer

    def __getitem__(self, index: int) -> BatchEncoding:
        element = self.elements[index]

        batch = self.tokenizer(
            element,
            truncation=True,
            max_length=512,  # TODO
            is_split_into_words=True,
            return_length=True,
        )

        length = batch["length"][0]
        del batch["length"]
        if self.tokenizer.truncation_side == "right":
            batch["context_mask"] = self._context_mask[index][:length]
        else:
            assert self.tokenizer.truncation_side == "left"
            batch["context_mask"] = self._context_mask[index][
                len(batch["input_ids"]) - length :
            ]

        return batch

    def __len__(self) -> int:
        return len(self.elements)


def ner_entities(
    tokens: List[str], bio_tags: List[str], resolve_inconsistencies: bool = True
) -> List[NEREntity]:
    """Extract NER entities from a list of BIO tags

    :param tokens: a list of tokens
    :param bio_tags: a list of BIO tags.  In particular, BIO tags
        should be in the CoNLL-2002 form (such as 'B-PER I-PER')

    :return: A list of ner entities, in apparition order
    """
    from renard.pipeline.ner import NEREntity

    assert len(tokens) == len(bio_tags)

    entities = []
    current_tag: Optional[str] = None
    current_tag_start_idx: Optional[int] = None

    for i, tag in enumerate(bio_tags):
        if not current_tag is None and not tag.startswith("I-"):
            assert not current_tag_start_idx is None
            entities.append(
                NEREntity(
                    tokens[current_tag_start_idx:i],
                    current_tag_start_idx,
                    i,
                    current_tag,
                )
            )
            current_tag = None
            current_tag_start_idx = None

        if tag.startswith("B-"):
            current_tag = tag[2:]
            current_tag_start_idx = i

        elif tag.startswith("I-"):
            if current_tag is None and resolve_inconsistencies:
                current_tag = tag[2:]
                current_tag_start_idx = i
                continue

    if not current_tag is None:
        assert not current_tag_start_idx is None
        entities.append(
            NEREntity(
                tokens[current_tag_start_idx : len(tokens)],
                current_tag_start_idx,
                len(bio_tags),
                current_tag,
            )
        )

    return entities


def load_conll2002_bio(
    path: str,
    tag_conversion_map: Optional[Dict[str, str]] = None,
    separator: str = "\t",
    max_sent_len: Optional[int] = None,
    **kwargs,
) -> Tuple[List[List[str]], List[str], List[NEREntity]]:
    """Load a file under CoNLL2022 BIO format.  Sentences are expected
    to be separated by end of lines.  Tags should be in the CoNLL-2002
    format (such as 'B-PER I-PER') - If this is not the case, see the
    ``tag_conversion_map`` argument.

    :param path: path to the CoNLL-2002 formatted file
    :param separator: separator between token and BIO tags
    :param tag_conversion_map: conversion map for tags found in the
        input file.  Example : ``{'B': 'B-PER', 'I': 'I-PER'}``
    :param max_sent_len: if specified, maximum length, in tokens, of
        sentences.
    :param kwargs: additional kwargs for :func:`open` (such as
        ``encoding`` or ``newline``).

    :return: ``(sentences, tokens, entities)``
    """
    tag_conversion_map = tag_conversion_map or {}

    with open(os.path.expanduser(path), **kwargs) as f:
        raw_data = f.read()

    sents = []
    sent_tokens = []
    tags = []
    for line in raw_data.split("\n"):
        line = line.strip("\n")
        if re.fullmatch(r"\s*", line) or (
            not max_sent_len is None and len(sent_tokens) >= max_sent_len
        ):
            if len(sent_tokens) == 0:
                continue
            sents.append(sent_tokens)
            sent_tokens = []
            continue
        token, tag = line.split(separator)
        sent_tokens.append(token)
        tags.append(tag_conversion_map.get(tag, tag))

    tokens = list(flatten(sents))
    entities = ner_entities(tokens, tags)

    return sents, list(flatten(sents)), entities


def hgdataset_from_conll2002(
    path: str,
    tag_conversion_map: Optional[Dict[str, str]] = None,
    separator: str = "\t",
    max_sent_len: Optional[int] = None,
    **kwargs,
) -> HGDataset:
    """Load a CoNLL-2002 file as a Huggingface Dataset.

    :param path: passed to :func:`.load_conll2002_bio`
    :param tag_conversion_map: passed to :func:`load_conll2002_bio`
    :param separator: passed to :func:`load_conll2002_bio`
    :param max_sent_len: passed to :func:`load_conll2002_bio`
    :param kwargs: additional kwargs for :func:`open`

    :return: a :class:`datasets.Dataset` with features 'tokens' and 'labels'.
    """
    sentences, tokens, entities = load_conll2002_bio(
        path, tag_conversion_map, separator, max_sent_len, **kwargs
    )

    # convert entities to labels
    tags = ["O"] * len(tokens)
    for entity in entities:
        entity_len = entity.end_idx - entity.start_idx
        tags[entity.start_idx : entity.end_idx] = [f"B-{entity.tag}"] + [
            f"I-{entity.tag}"
        ] * (entity_len - 1)

    # cut into sentences
    sent_ends = list(it.accumulate([len(s) for s in sentences]))
    sent_starts = [0] + sent_ends[:-1]
    sent_tags = [
        tags[sent_start:sent_end]
        for sent_start, sent_end in zip(sent_starts, sent_ends)
    ]

    dataset = HGDataset.from_dict({"tokens": sentences, "labels": sent_tags})
    dataset = dataset.cast_column(
        "labels", Sequence(ClassLabel(names=sorted(set(tags))))
    )
    return dataset


def _tokenize_and_align_labels(
    examples, tokenizer: PreTrainedTokenizerFast, label_all_tokens: bool = True
):
    """Adapted from https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/token_classification.ipynb#scrollTo=vc0BSBLIIrJQ

    :param examples: an object with keys 'tokens' and 'labels'
    """
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )

    labels = []
    for i, label in enumerate(examples[f"labels"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the
            # label to -100 so they are automatically ignored in the
            # loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            # For the other tokens in a word, we set the label to
            # either the current label or -100, depending on the
            # label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels

    return tokenized_inputs


def train_ner_model(
    hg_id: str,
    dataset: HGDataset,
    targs: TrainingArguments,
) -> PreTrainedModel:
    from transformers import DataCollatorForTokenClassification

    # BERT tokenizer splits tokens into subtokens. The
    # tokenize_and_align_labels function correctly aligns labels and
    # subtokens.
    tokenizer = AutoTokenizer.from_pretrained(hg_id)
    dataset = dataset.map(
        ft.partial(_tokenize_and_align_labels, tokenizer=tokenizer), batched=True
    )
    dataset = dataset.train_test_split(test_size=0.1)

    label_lst = dataset["train"].features["labels"].feature.names
    model = AutoModelForTokenClassification.from_pretrained(
        hg_id,
        num_labels=len(label_lst),
        id2label={i: label for i, label in enumerate(label_lst)},
        label2id={label: i for i, label in enumerate(label_lst)},
    )

    trainer = Trainer(
        model,
        targs,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        # data_collator=DataCollatorForTokenClassificationWithBatchEncoding(tokenizer),
        data_collator=DataCollatorForTokenClassification(tokenizer),
        tokenizer=tokenizer,
    )
    trainer.train()

    return model
