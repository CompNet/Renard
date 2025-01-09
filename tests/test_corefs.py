import os
import pytest
from renard.pipeline import Pipeline
from renard.pipeline.corefs import BertCoreferenceResolver


@pytest.mark.skipif(os.getenv("RENARD_TEST_ALL") != "1", reason="performance")
def test_bert_coreference_resolver_runs():
    pipeline = Pipeline([BertCoreferenceResolver()], progress_report=None)
    tokens = "Princess Liana felt sad , because Zarth Arn was gone . The princess went to sleep .".split(
        " "
    )
    corefs = pipeline(tokens=tokens).corefs
    assert not corefs is None
    assert len(corefs) >= 1
