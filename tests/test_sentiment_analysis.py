import unittest
from renard.pipeline.sentiment_analysis import NLTKSentimentAnalyzer


class TestNLTKSentimentAnalyzer(unittest.TestCase):
    """"""

    def setUp(self) -> None:
        self.sentiment_analyzer = NLTKSentimentAnalyzer()

    def test_polarity(self):
        pos_sent = "I love you .".split(" ")
        neg_sent = "I hate you !".split(" ")
        # text is not used, only sentences
        out_dict = self.sentiment_analyzer("", sentences=[pos_sent, neg_sent])

        pos_polarity = out_dict["sentences_polarities"][0]
        neg_polarity = out_dict["sentences_polarities"][1]
        self.assertGreaterEqual(pos_polarity, 0)
        self.assertLessEqual(neg_polarity, 0)
