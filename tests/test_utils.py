from renard.utils import spans


def test_known_input():
    assert set(
        [
            ("this",),
            ("this", "is"),
            ("is",),
            ("is", "a"),
            ("a",),
            ("a", "test"),
            ("test",),
        ]
    ) == set(spans("this is a test".split(" "), 2))
