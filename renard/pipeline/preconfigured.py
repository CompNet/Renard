from typing import Optional
from renard.pipeline.core import Pipeline
from renard.pipeline.tokenization import NLTKTokenizer
from renard.pipeline.character_unification import GraphRulesCharacterUnifier
from renard.pipeline.graph_extraction import (
    CoOccurrencesGraphExtractor,
    ConversationalGraphExtractor,
)
from renard.pipeline.quote_detection import QuoteDetector
from renard.pipeline.speaker_attribution import BertSpeakerDetector


def nltk_pipeline(
    tokenizer_kwargs: Optional[dict] = None,
    quote_detector_kwargs: Optional[dict] = None,
    ner_kwargs: Optional[dict] = None,
    characters_extractor_kwargs: Optional[dict] = None,
    speaker_detector_kwargs: Optional[dict] = None,
    graph_extractor_kwargs: Optional[dict] = None,
    conversational: bool = False,
    **pipeline_kwargs
) -> Pipeline:
    """A pre-configured NLTK-based pipeline

    :param tokenizer_kwargs: kwargs for :class:`.NLTKTokenizer`
    :param quote_detector_kwargs: kwargs for :class:`.QuoteDetector`
    :param ner_kwargs: kwargs for :class:`.BertNamedEntityRecognizer`
    :param characters_extractor_kwargs: kwargs for
        :class:`.GraphRulesCharacterUnifier`
    :param speaker_detector_kwargs: kwargs for :class:`.BertSpeakerDetector`
    :param graph_extractor_kwargs: kwargs for
        :class:`.CoOccurrencesGraphExtractor` or
        :class:`.ConversationalGraphExtractor`
    :param pipeline_kwargs: kwargs for :class:`.Pipeline`
    :param conversational: if ``True``, return a conversational pipeline
    """
    from renard.pipeline.ner import NLTKNamedEntityRecognizer

    tokenizer_kwargs = tokenizer_kwargs or {}
    ner_kwargs = ner_kwargs or {}
    characters_extractor_kwargs = characters_extractor_kwargs or {}
    graph_extractor_kwargs = graph_extractor_kwargs or {}

    if conversational:
        quote_detector_kwargs = quote_detector_kwargs or {}
        speaker_detector_kwargs = speaker_detector_kwargs or {}
        return Pipeline(
            [
                NLTKTokenizer(**tokenizer_kwargs),
                QuoteDetector(**quote_detector_kwargs),
                NLTKNamedEntityRecognizer(**ner_kwargs),
                GraphRulesCharacterUnifier(**characters_extractor_kwargs),
                BertSpeakerDetector(**speaker_detector_kwargs),
                ConversationalGraphExtractor(**graph_extractor_kwargs),
            ],
            **pipeline_kwargs
        )
    else:

        if not "co_occurrences_dist" in graph_extractor_kwargs:
            graph_extractor_kwargs["co_occurrences_dist"] = (1, "sentences")

        return Pipeline(
            [
                NLTKTokenizer(**tokenizer_kwargs),
                NLTKNamedEntityRecognizer(**ner_kwargs),
                GraphRulesCharacterUnifier(**characters_extractor_kwargs),
                CoOccurrencesGraphExtractor(**graph_extractor_kwargs),
            ],
            **pipeline_kwargs
        )


def bert_pipeline(
    tokenizer_kwargs: Optional[dict] = None,
    quote_detector_kwargs: Optional[dict] = None,
    ner_kwargs: Optional[dict] = None,
    characters_extractor_kwargs: Optional[dict] = None,
    speaker_detector_kwargs: Optional[dict] = None,
    graph_extractor_kwargs: Optional[dict] = None,
    conversational: bool = False,
    **pipeline_kwargs
) -> Pipeline:
    """A pre-configured BERT-based pipeline

    :param tokenizer_kwargs: kwargs for :class:`.NLTKTokenizer`
    :param quote_detector_kwargs: kwargs for :class:`.QuoteDetector`
    :param ner_kwargs: kwargs for :class:`.BertNamedEntityRecognizer`
    :param characters_extractor_kwargs: kwargs for
        :class:`.GraphRulesCharacterUnifier`
    :param speaker_detector_kwargs: kwargs for :class:`.BertSpeakerDetector`
    :param graph_extractor_kwargs: kwargs for
        :class:`.CoOccurrencesGraphExtractor` or
        :class:`.ConversationalGraphExtractor`
    :param pipeline_kwargs: kwargs for :class:`.Pipeline`
    :param conversational: if ``True``, return a conversational pipeline
    """
    from renard.pipeline.ner import BertNamedEntityRecognizer

    tokenizer_kwargs = tokenizer_kwargs or {}
    ner_kwargs = ner_kwargs or {}
    characters_extractor_kwargs = characters_extractor_kwargs or {}
    graph_extractor_kwargs = graph_extractor_kwargs or {}

    if conversational:
        quote_detector_kwargs = quote_detector_kwargs or {}
        speaker_detector_kwargs = speaker_detector_kwargs or {}
        return Pipeline(
            [
                NLTKTokenizer(**tokenizer_kwargs),
                QuoteDetector(**quote_detector_kwargs),
                BertNamedEntityRecognizer(**ner_kwargs),
                GraphRulesCharacterUnifier(**characters_extractor_kwargs),
                BertSpeakerDetector(**speaker_detector_kwargs),
                ConversationalGraphExtractor(**graph_extractor_kwargs),
            ],
            **pipeline_kwargs
        )
    else:

        if not "co_occurrences_dist" in graph_extractor_kwargs:
            graph_extractor_kwargs["co_occurrences_dist"] = (1, "sentences")

        return Pipeline(
            [
                NLTKTokenizer(**tokenizer_kwargs),
                BertNamedEntityRecognizer(**ner_kwargs),
                GraphRulesCharacterUnifier(**characters_extractor_kwargs),
                CoOccurrencesGraphExtractor(**graph_extractor_kwargs),
            ],
            **pipeline_kwargs
        )
