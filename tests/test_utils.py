import unittest
from renard.utils import spans


class TestSpans(unittest.TestCase):
    """"""

    def test_known_input(self):
        self.assertEqual(
            set(
                [
                    ("this",),
                    ("this", "is"),
                    ("is",),
                    ("is", "a"),
                    ("a",),
                    ("a", "test"),
                    ("test",),
                ]
            ),
            set(spans("this is a test".split(" "), 2)),
        )


if __name__ == "__main__":
    unittest.main()
