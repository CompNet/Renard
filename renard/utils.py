from more_itertools.more import windowed
from typing import List, Tuple, TypeVar, Iterable, cast


T = TypeVar("T")


def spans(seq: Iterable[T], max_len: int) -> List[Tuple[T]]:
    """Cut the input sequence into all possible spans up to a maximum length

    .. note::

        spans are ordered from the smallest to the biggest,
        from the beginning of seq to the end of seq.


    :param seq:
    :param max_len:
    :return:
    """
    out_spans = []
    for i in range(1, max_len + 1):
        for span in windowed(seq, i):
            out_spans.append(span)
    return out_spans


def spans_indexs(seq: List, max_len: int) -> List[Tuple[int, int]]:
    """"""
    indexs = []
    for i in range(1, max_len + 1):
        for span in windowed(range(len(seq)), i):
            span = cast(Tuple[int, ...], span)
            indexs.append((min(span), max(span)))
    return indexs
