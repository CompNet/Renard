import os
import pytest
from renard.pipeline.core import Pipeline
from renard.resources.novels import load_novel_chapters

script_dir = os.path.abspath(os.path.dirname(__file__))


@pytest.mark.skipif(
    os.getenv("RENARD_TEST_OPTDEP_STANZA") != "1",
    reason="not testing stanza based modules",
)
def test_stanza_pipeline_runs():
    from renard.pipeline.stanford_corenlp import StanfordCoreNLPPipeline

    text = load_novel_chapters("pride_and_prejudice")[0]
    text = text[:1000]  # limit size for performance reasons
    pipeline = Pipeline([StanfordCoreNLPPipeline()], warn=False, progress_report=None)
    out = pipeline(text)
    assert out.entities and len(out.entities) > 0
