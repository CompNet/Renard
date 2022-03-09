from more_itertools.more import windowed
from typing import List, Tuple, TypeVar, Iterable


T = TypeVar("T")


def spans(seq: Iterable[T], max_len: int) -> List[Tuple[T]]:
    """
    :param seq:
    :param max_len:
    :return:
    """
    out_spans = []
    for i in range(1, max_len + 1):
        for span in windowed(seq, i):
            out_spans.append(span)
    return out_spans
