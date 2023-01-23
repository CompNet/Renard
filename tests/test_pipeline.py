from typing import Set
import os
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st
from renard.pipeline.core import Pipeline, PipelineStep
from renard.pipeline.preconfigured import bert_pipeline, nltk_pipeline


script_dir = os.path.abspath(os.path.dirname(__file__))


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
def test_nltk_pipeline_runs():
    with open(f"{script_dir}/pp_chapter1.txt") as f:
        text = f.read()
    pipeline = nltk_pipeline(warn=False, progress_report=None)
    pipeline(text)


@pytest.mark.skipif(os.getenv("RENARD_TEST_ALL") != "1", reason="performance")
def test_bert_pipeline_runs():
    with open(f"{script_dir}/pp_chapter1.txt") as f:
        text = f.read()
    pipeline = bert_pipeline(warn=False, progress_report=None)
    pipeline(text)
