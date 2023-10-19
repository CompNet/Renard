from renard.pipeline import Mention
from renard.pipeline.character_unification import (
    Character,
    _assign_coreference_mentions,
)


def test_assign_coreference_mentions():
    characters = _assign_coreference_mentions(
        [Character(frozenset(["John Traitor"]), [Mention(["John", "Traitor"], 0, 1)])],
        [[Mention(["John", "Traitor"], 0, 1), Mention(["He"], 10, 10)]],
    )
    assert characters[0] == Character(
        frozenset(["John Traitor"]),
        [Mention(["John", "Traitor"], 0, 1), Mention(["He"], 10, 10)],
    )
