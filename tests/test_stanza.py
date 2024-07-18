import os
import pytest
from renard.pipeline.core import Pipeline

script_dir = os.path.abspath(os.path.dirname(__file__))


@pytest.mark.skipif(
    os.getenv("RENARD_TEST_STANZA_OPTDEP") != "1",
    reason="not testing stanza based modules",
)
def test_stanza_pipeline_runs():
    from renard.pipeline.stanford_corenlp import StanfordCoreNLPPipeline

    with open(f"{script_dir}/pp_chapter1.txt") as f:
        text = f.read()
    text = text[:1000]  # limit size for performance reasons
    pipeline = Pipeline([StanfordCoreNLPPipeline()], warn=False, progress_report=None)
    out = pipeline(text)
    assert out.entities and len(out.entities) > 0
