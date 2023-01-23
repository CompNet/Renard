from typing import Set
import os
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st
from renard.pipeline.core import Pipeline, PipelineStep
from renard.pipeline.tokenization import NLTKTokenizer
from renard.pipeline.ner import NLTKNamedEntityRecognizer
from renard.pipeline.sentiment_analysis import NLTKSentimentAnalyzer
from renard.pipeline.characters_extraction import NaiveCharactersExtractor
from renard.pipeline.graph_extraction import CoOccurencesGraphExtractor


def test_pipeline_is_valid():
    class TestPipelineStep1(PipelineStep):
        def needs(self) -> Set[str]:
            return set()

        def production(self) -> Set[str]:
            return {"info_1"}

    class TestPipelineStep2(PipelineStep):
        def needs(self) -> Set[str]:
            return {"info_1"}

        def production(self) -> Set[str]:
            return set()

    pipeline = Pipeline([TestPipelineStep1(), TestPipelineStep2()])

    assert pipeline.check_valid()[0]


def test_pipeline_is_invalid():
    class TestPipelineStep1(PipelineStep):
        def needs(self) -> Set[str]:
            return set()

        def production(self) -> Set[str]:
            return set()

    class TestPipelineStep2(PipelineStep):
        def needs(self) -> Set[str]:
            return {"info_1"}

        def production(self) -> Set[str]:
            return set()

    pipeline = Pipeline([TestPipelineStep1(), TestPipelineStep2()])

    assert not pipeline.check_valid()[0]


@pytest.mark.skipif(os.getenv("RENARD_TEST_ALL") != "1", reason="performance")
@settings(max_examples=25, deadline=None)
@given(text=st.text())
def test_nltk_pipeline(text: str):
    pipeline = Pipeline(
        [
            NLTKTokenizer(),
            NLTKNamedEntityRecognizer(),
            NLTKSentimentAnalyzer(),
            NaiveCharactersExtractor(),
            CoOccurencesGraphExtractor(co_occurences_dist=10),
        ],
        warn=False,
        progress_report=None,
    )
    pipeline(text)
