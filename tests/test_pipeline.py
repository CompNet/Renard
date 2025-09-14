from typing import Set
import os
import pytest
from renard.pipeline.core import Pipeline, PipelineStep
from renard.pipeline.preconfigured import (
    co_occurrence_pipeline,
    conversational_pipeline,
    relational_pipeline,
)
from renard.resources.novels.novels import load_novel_chapters


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
def test_co_occurrence_pipeline_runs():
    text = load_novel_chapters("pride_and_prejudice")[0]
    pipeline = co_occurrence_pipeline(warn=False, progress_report=None)
    pipeline(text)


@pytest.mark.skipif(os.getenv("RENARD_TEST_ALL") != "1", reason="performance")
def test_conversational_pipeline_runs():
    text = load_novel_chapters("pride_and_prejudice")[0]
    # # if the text is too long, speaker attribution takes too much time
    # text = text[:500]

    pipeline = conversational_pipeline(warn=False, progress_report=None)
    pipeline(text)


@pytest.mark.skipif(os.getenv("RENARD_TEST_ALL") != "1", reason="performance")
def test_relational_pipeline_runs():
    text = load_novel_chapters("pride_and_prejudice")[0]
    pipeline = relational_pipeline(warn=False, progress_report=None)
    pipeline(text)
