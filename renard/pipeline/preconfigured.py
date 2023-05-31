from renard.pipeline.core import Pipeline
from renard.pipeline.characters_extraction import GraphRulesCharactersExtractor
from renard.pipeline.graph_extraction import CoOccurrencesGraphExtractor


def nltk_pipeline(**kwargs) -> Pipeline:
    from renard.pipeline.tokenization import NLTKTokenizer
    from renard.pipeline.ner import NLTKNamedEntityRecognizer
    from renard.pipeline.sentiment_analysis import NLTKSentimentAnalyzer

    return Pipeline(
        [
            NLTKTokenizer(),
            NLTKNamedEntityRecognizer(),
            NLTKSentimentAnalyzer(),
            GraphRulesCharactersExtractor(),
            CoOccurrencesGraphExtractor(co_occurences_dist=(1, "sentences")),
        ],
        **kwargs
    )


def bert_pipeline(**kwargs) -> Pipeline:
    from renard.pipeline.tokenization import BertTokenizer
    from renard.pipeline.ner import BertNamedEntityRecognizer

    return Pipeline(
        [
            BertTokenizer(),
            BertNamedEntityRecognizer(),
            GraphRulesCharactersExtractor(),
            CoOccurrencesGraphExtractor(co_occurences_dist=(1, "sentences")),
        ],
        **kwargs
    )
