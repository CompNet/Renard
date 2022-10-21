import unittest
from renard.pipeline import Mention
from renard.pipeline.characters_extraction import (
    Character,
    _assign_coreference_mentions,
)


class TestCoreferenceMentionsAssignment(unittest.TestCase):
    def test_assign_coreference_mentions(self):
        characters = _assign_coreference_mentions(
            [
                Character(
                    frozenset(["John Traitor"]), [Mention(["John", "Traitor"], 0, 1)]
                )
            ],
            [[Mention(["John", "Traitor"], 0, 1), Mention(["He"], 10, 10)]],
        )
        assert characters[0] == Character(
            frozenset(["John Traitor"]),
            [Mention(["John", "Traitor"], 0, 1), Mention(["He"], 10, 10)],
        )


if __name__ == "__main__":
    unittest.main()
