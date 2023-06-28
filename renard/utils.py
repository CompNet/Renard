from typing import List, Tuple, TypeVar, Collection, Iterable, Optional, Dict, cast
import re, os
from more_itertools import flatten
from more_itertools.more import windowed
import torch

from renard.pipeline.ner import NEREntity, ner_entities


T = TypeVar("T")


def spans(seq: Collection[T], max_len: int) -> List[Tuple[T]]:
    """Cut the input sequence into all possible spans up to a maximum length

    .. note::

        spans are ordered from the smallest to the biggest,
        from the beginning of seq to the end of seq.


    :param seq:
    :param max_len:
    :return:
    """
    out_spans = []
    for i in range(1, min(len(seq), max_len + 1)):
        for span in windowed(seq, i):
            out_spans.append(span)
    return out_spans


def spans_indexs(seq: List, max_len: int) -> List[Tuple[int, int]]:
    """"""
    indexs = []
    for i in range(1, min(len(seq), max_len + 1)):
        for span in windowed(range(len(seq)), i):
            span = cast(Tuple[int, ...], span)
            indexs.append((min(span), max(span)))
    return indexs


def batch_index_select(
    input: torch.Tensor, dim: int, index: torch.Tensor
) -> torch.Tensor:
    """Batched version of :func:`torch.index_select`.
    Inspired by https://discuss.pytorch.org/t/batched-index-select/9115/8

    :param input: a torch tensor of shape ``(B, *)`` where ``*``
        is any number of additional dimensions.
    :param dim: the dimension in which to index
    :param index: index tensor of shape ``(B, I)``

    :return: a tensor which indexes ``input`` along dimension ``dim``
        using ``index``. This tensor has the same shape as ``input``,
        except in dimension ``dim``, where it has dimension ``I``.
    """
    batch_size = input.shape[0]

    view = [batch_size] + [1 if i != dim else -1 for i in range(1, len(input.shape))]

    expansion = list(input.shape)
    expansion[0] = batch_size
    expansion[dim] = -1

    return torch.gather(input, dim, index.view(view).expand(expansion))


R = TypeVar("R")


def search_pattern(seq: Iterable[R], pattern: List[R]) -> List[int]:
    """Search a pattern in sequence

    :param seq: sequence in which to search
    :param pattern: searched pattern
    :return: a list of patterns start index
    """
    start_indices = []
    for subseq_i, subseq in enumerate(windowed(seq, len(pattern))):
        if list(subseq) == pattern:
            start_indices.append(subseq_i)
    return start_indices


def load_conll2002_bio(
    path: str,
    tag_conversion_map: Optional[Dict[str, str]] = None,
    separator: str = "\t",
    **kwargs
) -> Tuple[List[List[str]], List[str], List[NEREntity]]:
    """Load a file under CoNLL2022 BIO format.  Sentences are expected
    to be separated by end of lines.  Tags should be in the CoNLL-2002
    format (such as 'B-PER I-PER') - If this is not the case, see the
    ``tag_conversion_map`` argument.

    :param path: path to the CoNLL-2002 formatted file
    :param separator: separator between token and BIO tags
    :param tag_conversion_map: conversion map for tags found in the
        input file.  Example : ``{'B': 'B-PER', 'I': 'I-PER'}``
    :param kwargs: additional kwargs for ``open`` (such as
        ``encoding`` or ``newline``).

    :return: ``(sentences, tokens, entities)``
    """

    if tag_conversion_map is None:
        tag_conversion_map = {}

    with open(os.path.expanduser(path), **kwargs) as f:
        raw_data = f.read()

    sents = []
    sent_tokens = []
    tags = []
    for line in raw_data.split("\n"):
        line = line.strip("\n")
        if re.fullmatch(r"\s*", line):
            sents.append(sent_tokens)
            sent_tokens = []
            continue
        token, tag = line.split(separator)
        sent_tokens.append(token)
        tags.append(tag_conversion_map.get(tag, tag))

    tokens = list(flatten(sents))
    entities = ner_entities(tokens, tags)

    return sents, list(flatten(sents)), entities
