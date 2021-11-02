from typing import Iterable, Generator, TypeVar, Optional
from itertools import islice


T = TypeVar("T")


def sliding_window(
    seq: Iterable[T], n: int = 2, padding: int = 0
) -> Generator[Optional[T], None, None]:
    """
    Sliding window generator over a sequence

    :param seq: input sequence
    :param n: sliding window size
    :param padding: sliding window padding size (None will be used to pad)
    """
    if n <= 0:
        raise Exception(f"wrong value of n : {n}")

    iterator = iter(seq)
    window = [None] * padding + list(islice(iterator, n - padding))  # type: ignore
    end_padding = padding + len(window) - n

    if len(window) < n:
        window += [None] * (n - len(window))

    if len(window) == n:
        yield window

    for elem in iterator:
        window = window[1:] + [elem]
        yield window

    for _ in range(end_padding):
        window = window[1:] + [None]
        yield window
