from typing import Set
import unittest
from hypothesis import given, settings
import hypothesis.strategies as st
from renard.pipeline.core import Pipeline, PipelineStep
from renard.pipeline.tokenization import NLTKTokenizer
from renard.pipeline.ner import NLTKNamedEntityRecognizer
from renard.pipeline.sentiment_analysis import NLTKSentimentAnalyzer
from renard.pipeline.characters_extraction import NaiveCharactersExtractor
from renard.pipeline.graph_extraction import CoOccurencesGraphExtractor


class TestPipelineValidity(unittest.TestCase):
    """"""

    def test_pipeline_is_valid(self):
        """"""

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

        pipeline = Pipeline((TestPipelineStep1(), TestPipelineStep2()))

        self.assertTrue(pipeline.check_valid()[0])

    def test_pipeline_is_invalid(self):
        """"""

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

        pipeline = Pipeline((TestPipelineStep1(), TestPipelineStep2()))

        self.assertFalse(pipeline.check_valid()[0])


class TestCompleteNLTKPipeline(unittest.TestCase):
    """"""

    def setUp(self) -> None:
        self.pipeline = Pipeline(
            [
                NLTKTokenizer(),
                NLTKNamedEntityRecognizer(),
                NLTKSentimentAnalyzer(),
                NaiveCharactersExtractor(),
                CoOccurencesGraphExtractor(co_occurences_dist=10),
            ],
            progress_report=None,
            warn=False,
        )

    @settings(max_examples=25)
    @given(text=st.text())
    def test_nltk_pipeline(self, text: str):
        self.pipeline(text)


if __name__ == "__main__":
    unittest.main()
