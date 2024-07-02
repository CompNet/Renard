from __future__ import annotations
from typing import Iterable, Literal, Optional, TypeVar, Generator
import sys
from tqdm import tqdm


class ProgressReporter:
    """
    An abstract class representing an object that reports a pipeline
    step progress.
    """

    def start_(self, total: int):
        """Method called at step start time."""
        self.total = total

    def update_progress_(self, added_progress: int):
        """Update step progress."""
        raise NotImplementedError

    def update_message_(self, message: str):
        """Update reporter current message."""
        pass

    def get_subreporter(self) -> ProgressReporter:
        """Get the subreporter corresponding to that reporter."""
        raise NotImplementedError


class NoopProgressReporter(ProgressReporter):
    def reset_(self):
        pass

    def update_progress_(self, added_progress: int):
        pass

    def get_subreporter(self) -> ProgressReporter:
        return NoopProgressReporter()


class TQDMSubProgressReporter(ProgressReporter):
    def __init__(self, reporter: TQDMProgressReporter) -> None:
        super().__init__()
        self.reporter = reporter

    def start_(self, total: int):
        super().start_(total)
        self.progress = 0

    def update_progress_(self, added_progress: int):
        self.progress += added_progress
        self.reporter.tqdm.set_postfix(step=f"({self.progress}/{self.total})")

    def update_message_(self, message: str):
        self.reporter.tqdm.set_postfix(
            step=f"({self.progress}/{self.total})", message=message
        )


class TQDMProgressReporter(ProgressReporter):
    def start_(self, total: int):
        super().start_(total)
        self.tqdm = tqdm(total=total)

    def update_progress_(self, added_progress: int):
        self.tqdm.update(added_progress)

    def update_message_(self, message: str):
        self.tqdm.set_description_str(message)

    def get_subreporter(self) -> ProgressReporter:
        return TQDMSubProgressReporter(self)


T = TypeVar("T")


def progress_(
    progress_reporter: ProgressReporter,
    it: Iterable[T],
    total: Optional[int] = None,
) -> Generator[T, None, None]:
    if total == None:
        total = len(it)  # type: ignore
    progress_reporter.start_(total)
    for elt in it:
        progress_reporter.update_progress_(1)
        yield elt


def get_progress_reporter(name: Optional[Literal["tqdm"]]) -> ProgressReporter:
    if name is None:
        return NoopProgressReporter()
    if name == "tqdm":
        return TQDMProgressReporter()
    print(f"[warning] unknown progress reporter: {name}", file=sys.stderr)
    return NoopProgressReporter()
