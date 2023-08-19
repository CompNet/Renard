from typing import Optional
from renard.pipeline.core import Pipeline
from renard.pipeline.characters_extraction import GraphRulesCharactersExtractor
from renard.pipeline.graph_extraction import CoOccurrencesGraphExtractor


def nltk_pipeline(
    tokenizer_kwargs: Optional[dict] = None,
    ner_kwargs: Optional[dict] = None,
    characters_extractor_kwargs: Optional[dict] = None,
    graph_extractor_kwargs: Optional[dict] = None,
    **pipeline_kwargs
) -> Pipeline:
    """A pre-configured NLTK-based pipeline

    :param tokenizer_kwargs: kwargs for :class:`.NLTKTokenizer`
    :param ner_kwargs: kwargs for :class:`.NLTKNamedEntityRecognizer`
    :param characters_extractor_kwargs: kwargs for :class:`.GraphRulesCharactersExtractor`
    :param graph_extractor_kwargs: kwargs for :class:`.CoOccurrencesGraphExtractor`
    :param pipeline_kwargs: kwargs for :class:`.Pipeline`
    """
    from renard.pipeline.tokenization import NLTKTokenizer
    from renard.pipeline.ner import NLTKNamedEntityRecognizer

    tokenizer_kwargs = tokenizer_kwargs or {}
    ner_kwargs = ner_kwargs or {}
    characters_extractor_kwargs = characters_extractor_kwargs or {}
    graph_extractor_kwargs = graph_extractor_kwargs or {}

    if not "co_occurences_dist" in graph_extractor_kwargs:
        graph_extractor_kwargs["co_occurences_dist"] = (1, "sentences")

    return Pipeline(
        [
            NLTKTokenizer(**tokenizer_kwargs),
            NLTKNamedEntityRecognizer(**ner_kwargs),
            GraphRulesCharactersExtractor(**characters_extractor_kwargs),
            CoOccurrencesGraphExtractor(**graph_extractor_kwargs),
        ],
        **pipeline_kwargs
    )


def bert_pipeline(
    tokenizer_kwargs: Optional[dict] = None,
    ner_kwargs: Optional[dict] = None,
    characters_extractor_kwargs: Optional[dict] = None,
    graph_extractor_kwargs: Optional[dict] = None,
    **pipeline_kwargs
) -> Pipeline:
    """A pre-configured BERT-based pipeline

    :param tokenizer_kwargs: kwargs for :class:`.NLTKTokenizer`
    :param ner_kwargs: kwargs for :class:`.BertNamedEntityRecognizer`
    :param characters_extractor_kwargs: kwargs for :class:`.GraphRulesCharactersExtractor`
    :param graph_extractor_kwargs: kwargs for :class:`.CoOccurrencesGraphExtractor`
    :param pipeline_kwargs: kwargs for :class:`.Pipeline`
    """
    from renard.pipeline.tokenization import NLTKTokenizer
    from renard.pipeline.ner import BertNamedEntityRecognizer

    tokenizer_kwargs = tokenizer_kwargs or {}
    ner_kwargs = ner_kwargs or {}
    characters_extractor_kwargs = characters_extractor_kwargs or {}
    graph_extractor_kwargs = graph_extractor_kwargs or {}

    return Pipeline(
        [
            NLTKTokenizer(),
            BertNamedEntityRecognizer(),
            GraphRulesCharactersExtractor(),
            CoOccurrencesGraphExtractor(co_occurences_dist=(1, "sentences")),
        ],
        **pipeline_kwargs
    )
