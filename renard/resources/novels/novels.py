import re
import pathlib as pl

NOVELS = [
    path.name
    for path in pl.Path(__file__).parent.iterdir()
    if path.is_dir() and path.name != "__pycache__"
]


def load_novel_chapters(name: str) -> list[str]:
    chapters = []
    chapter_paths = (pl.Path(__file__).parent / name).glob("chapter_*.txt")
    chapter_paths = sorted(
        chapter_paths,
        key=lambda p: int(re.match(r"chapter_([0-9]+)\.txt", str(p.name)).group(1)),
    )
    for path in sorted((pl.Path(__file__).parent / name).glob("chapter_*.txt")):
        with open(path) as f:
            chapters.append(f.read())
    return chapters


def load_novel(name: str) -> str:
    return "\n".join(load_novel_chapters(name))
