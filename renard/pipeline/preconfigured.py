from typing import Optional, Type
from renard.pipeline.core import Pipeline, PipelineStep
from renard.pipeline.ner.ner import NLTKNamedEntityRecognizer


def co_occurrence_pipeline(
    tokenizer_kwargs: Optional[dict] = None,
    ner_step: Type[PipelineStep] = NLTKNamedEntityRecognizer,
    ner_kwargs: Optional[dict] = None,
    character_unifier_kwargs: Optional[dict] = None,
    graph_extractor_kwargs: Optional[dict] = None,
    **pipeline_kwargs,
) -> Pipeline:
    """Return a pre-configured co-occurrence pipeline.

    :param tokenizer_kwargs: kwargs for :class:`.NLTKTokenizer`
    :param ner_step: the class of the NER step to use.
    :param ner_kwargs: kwargs for :class:`.BertNamedEntityRecognizer`
    :param character_unifier_kwargs: kwargs for
        :class:`.GraphRulesCharacterUnifier`
    :param graph_extractor_kwargs: kwargs for
        :class:`.CoOccurrencesGraphExtractor`,
        :class:`.ConversationalGraphExtractor` or
        :class:`.RelationalGraphExtractor`
    :param pipeline_kwargs: kwargs for :class:`.Pipeline`
    """
    from renard.pipeline.tokenization import NLTKTokenizer
    from renard.pipeline.character_unification import GraphRulesCharacterUnifier
    from renard.pipeline.graph_extraction import CoOccurrencesGraphExtractor

    tokenizer_kwargs = tokenizer_kwargs or {}
    ner_kwargs = ner_kwargs or {}
    character_unifier_kwargs = character_unifier_kwargs or {}
    graph_extractor_kwargs = graph_extractor_kwargs or {}

    if not "co_occurrences_dist" in graph_extractor_kwargs:
        graph_extractor_kwargs["co_occurrences_dist"] = (1, "sentences")

    return Pipeline(
        [
            NLTKTokenizer(**tokenizer_kwargs),
            ner_step(**ner_kwargs),
            GraphRulesCharacterUnifier(**character_unifier_kwargs),
            CoOccurrencesGraphExtractor(**graph_extractor_kwargs),
        ],
        **pipeline_kwargs,
    )


def conversational_pipeline(
    tokenizer_kwargs: Optional[dict] = None,
    ner_step: Type[PipelineStep] = NLTKNamedEntityRecognizer,
    ner_kwargs: Optional[dict] = None,
    character_unifier_kwargs: Optional[dict] = None,
    quote_detector_kwargs: Optional[dict] = None,
    speaker_detector_kwargs: Optional[dict] = None,
    graph_extractor_kwargs: Optional[dict] = None,
    **pipeline_kwargs,
):
    """Return a preconfigured conversational extraction pipeline.

    :param tokenizer_kwargs: kwargs for :class:`.NLTKTokenizer`
    :param ner_step: the type of the NER step to use.
    :param ner_kwargs: kwargs for :class:`.BertNamedEntityRecognizer`
    :param character_unifier_kwargs: kwargs for
        :class:`.GraphRulesCharacterUnifier`
    :param quote_detector_kwargs: kwargs for :class:`.QuoteDetector`
    :param speaker_detector_kwargs: kwargs for
        :class:`.BertSpeakerDetector` in the case of a conversational
        pipeline.
    :param graph_extractor_kwargs: kwargs for
        :class:`.ConversationalGraphExtractor`
    :param pipeline_kwargs: kwargs for :class:`.Pipeline`
    """
    from renard.pipeline.tokenization import NLTKTokenizer
    from renard.pipeline.character_unification import GraphRulesCharacterUnifier
    from renard.pipeline.graph_extraction import ConversationalGraphExtractor
    from renard.pipeline.quote_detection import QuoteDetector
    from renard.pipeline.speaker_attribution import BertSpeakerDetector

    tokenizer_kwargs = tokenizer_kwargs or {}
    ner_kwargs = ner_kwargs or {}
    character_unifier_kwargs = character_unifier_kwargs or {}
    quote_detector_kwargs = quote_detector_kwargs or {}
    speaker_detector_kwargs = speaker_detector_kwargs or {}
    graph_extractor_kwargs = graph_extractor_kwargs or {}

    if not "graph_type" in graph_extractor_kwargs:
        graph_extractor_kwargs["graph_type"] = "conversation"
    if (
        graph_extractor_kwargs["graph_type"] == "conversation"
        and not "conversation_dist" in graph_extractor_kwargs
    ):
        graph_extractor_kwargs["conversation_dist"] = 1

    return Pipeline(
        [
            NLTKTokenizer(**tokenizer_kwargs),
            QuoteDetector(**quote_detector_kwargs),
            ner_step(**ner_kwargs),
            GraphRulesCharacterUnifier(**character_unifier_kwargs),
            BertSpeakerDetector(**speaker_detector_kwargs),
            ConversationalGraphExtractor(**graph_extractor_kwargs),
        ],
        **pipeline_kwargs,
    )


def relational_pipeline(
    tokenizer_kwargs: Optional[dict] = None,
    ner_step: Type[PipelineStep] = NLTKNamedEntityRecognizer,
    ner_kwargs: Optional[dict] = None,
    character_unifier_kwargs: Optional[dict] = None,
    relation_extractor_kwargs: Optional[dict] = None,
    graph_extractor_kwargs: Optional[dict] = None,
    **pipeline_kwargs,
):
    """Return a preconfigured relational extraction pipeline.

    :param tokenizer_kwargs: kwargs for :class:`.NLTKTokenizer`
    :param ner_step: the type of the NER step to use.
    :param ner_kwargs: kwargs for :class:`.BertNamedEntityRecognizer`
    :param characters_unifier_kwargs: kwargs for
        :class:`.GraphRulesCharacterUnifier`
    :param relation_extraction_kwargs: kwargs for
        :class:`.GenerativeRelationExtractor`
    :param pipeline_kwargs: kwargs for :class:`.Pipeline`
    """
    from renard.pipeline.tokenization import NLTKTokenizer
    from renard.pipeline.character_unification import GraphRulesCharacterUnifier
    from renard.pipeline.relation_extraction import GenerativeRelationExtractor
    from renard.pipeline.graph_extraction import RelationalGraphExtractor

    tokenizer_kwargs = tokenizer_kwargs or {}
    ner_kwargs = ner_kwargs or {}
    character_unifier_kwargs = character_unifier_kwargs or {}
    relation_extractor_kwargs = relation_extractor_kwargs or {}
    graph_extractor_kwargs = graph_extractor_kwargs or {}

    return Pipeline(
        [
            NLTKTokenizer(**tokenizer_kwargs),
            ner_step(**ner_kwargs),
            GraphRulesCharacterUnifier(**character_unifier_kwargs),
            GenerativeRelationExtractor(**relation_extractor_kwargs),
            RelationalGraphExtractor(**graph_extractor_kwargs),
        ],
        **pipeline_kwargs,
    )
