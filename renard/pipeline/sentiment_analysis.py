from typing import Dict, Any, List, Set, Optional
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from renard.pipeline.core import PipelineStep


class NLTKSentimentAnalyzer(PipelineStep):
    """A sentiment analyzer based on NLTK's Vader.

    Hutto, C.J. & Gilbert, E.E. (2014).  VADER: A Parsimonious
    Rule-based Model for Sentiment Analysis of Social Media Text.
    Eighth International Conference on Weblogs and Social Media
    (ICWSM-14).  Ann Arbor, MI, June 2014.
    """

    def __init__(self) -> None:
        nltk.download("vader_lexicon", quiet=True)
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        super().__init__()

    def __call__(self, sentences: List[List[str]], **kwargs) -> Dict[str, Any]:
        return {
            # TODO: here, we 'detokenized' sentences using a very
            # naive method. This probably does not matter for Vader,
            # but it could be better. A solution could be keeping raw
            # sentences in the pipeline as 'sentences', while
            # 'sentences_tokens' would replace the current 'sentences'
            # attribute.
            "sentences_polarities": [
                self.sentiment_analyzer.polarity_scores(" ".join(s))["compound"]
                for s in sentences
            ]
        }

    def needs(self) -> Set[str]:
        return {"sentences"}

    def production(self) -> Set[str]:
        return {"sentences_polarities"}
