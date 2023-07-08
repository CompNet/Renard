import os
import pytest
from renard.pipeline.corefs import BertCoreferenceResolver, Mention


@pytest.mark.skipif(os.getenv("RENARD_TEST_ALL") != "1", reason="performance")
def test_bert_coreference_resolver_runs():
    coref_step = BertCoreferenceResolver()
    coref_step._pipeline_init_("eng", None)  # type: ignore
    tokens = "Princess Liana felt sad , because Zarth Arn was gone . The princess went to sleep .".split(
        " "
    )
    corefs = coref_step(tokens)["corefs"]
    assert len(corefs) == 2
