from typing import Set
import unittest
from renard.pipeline.core import Pipeline, PipelineStep


class TestPipelineValidity(unittest.TestCase):
    """"""

    def test_pipeline_is_valid(self):
        """"""

        class TestPipelineStep1(PipelineStep):
            def needs(self) -> Set[str]:
                return set()

            def produces(self) -> Set[str]:
                return {"info_1"}

        class TestPipelineStep2(PipelineStep):
            def needs(self) -> Set[str]:
                return {"info_1"}

            def produces(self) -> Set[str]:
                return set()

        pipeline = Pipeline((TestPipelineStep1(), TestPipelineStep2()))

        self.assertTrue(pipeline.check_valid()[0])

    def test_pipeline_is_invalid(self):
        """"""

        class TestPipelineStep1(PipelineStep):
            def needs(self) -> Set[str]:
                return set()

            def produces(self) -> Set[str]:
                return set()

        class TestPipelineStep2(PipelineStep):
            def needs(self) -> Set[str]:
                return {"info_1"}

            def produces(self) -> Set[str]:
                return set()

        pipeline = Pipeline((TestPipelineStep1(), TestPipelineStep2()))

        self.assertFalse(pipeline.check_valid()[0])
