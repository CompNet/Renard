from renard.pipeline.sentiment_analysis import NLTKSentimentAnalyzer


def test_polarity():
    sentiment_analyzer = NLTKSentimentAnalyzer()
    pos_sent = "I love you .".split(" ")
    neg_sent = "I hate you !".split(" ")
    # text is not used, only sentences
    out_dict = sentiment_analyzer(sentences=[pos_sent, neg_sent])

    pos_polarity = out_dict["sentences_polarities"][0]
    neg_polarity = out_dict["sentences_polarities"][1]
    assert pos_polarity >= 0
    assert neg_polarity <= 0
