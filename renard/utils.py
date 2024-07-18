from typing import List, Literal, Tuple, TypeVar, Collection, Iterable, cast, Union
import sys
from more_itertools.more import windowed
import torch

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


#: A `BlockBounds` delimits blocks in either raw text ("characters") or
#: tokenized text ("tokens"). It has the following form:
#:
#: ([(block start, block end), ...], unit)
#:
#: see :func:`block_indices` to easily create `BlockBounds`
BlockBounds = Tuple[List[Tuple[int, int]], Literal["characters", "tokens"]]


def block_bounds(blocks: Union[List[str], List[List[str]]]) -> BlockBounds:
    """Return the boundaries of a series of blocks.

    :param blocks: either a list of raw texts or a list of tokenized
        texts.

    :return: A `BlockBounds` with the correct unit.
    """
    if len(blocks) == 0:
        print("[warning] computing block bounds on 0 blocks.", file=sys.stderr)
        return ([], ("characters"))

    if isinstance(blocks[0], str):
        unit = "characters"
    elif isinstance(blocks[0], list):
        unit = "tokens"
    else:
        raise ValueError(blocks)

    indices = []
    start = 0
    for block in blocks:
        end = start + len(block)
        indices.append((start, end))
        start = end

    return (indices, unit)


def charbb2tokenbb(char_bb: BlockBounds, char2token: List[int]) -> BlockBounds:
    """Convert a `BlockBounds` in characters to a `BlockBounds` in
    tokens.

    :param char_bb: block bounds, in 'characters'.
    :param char2token: a list with ``char2token[i]`` being the index
        of token corresponding to character ``i``.

    :return: a `BlockBounds`, in 'tokens'.
    """
    assert char_bb[1] == "characters"
    tokens_blocks = []
    for char_block_start, char_block_end in char_bb[0]:
        tokens_blocks.append((char2token[char_block_start], char2token[char_block_end]))
    return (tokens_blocks, "tokens")
