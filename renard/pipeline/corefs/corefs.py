from typing import List, Set, Dict, Any, cast
import torch
from transformers import BertTokenizerFast  # type: ignore
from renard.pipeline import PipelineStep
from renard.pipeline.corefs.mentions import CoreferenceMention
from renard.pipeline.corefs.bert_corefs import BertForCoreferenceResolution


class BertCoreferenceResolver(PipelineStep):
    """
    A coreference resolver based using BERT.  Loosely based on
    'End-to-end Neural Coreference Resolution' (Lee et at.  2017) and
    'BERT for coreference resolution' (Joshi et al.  2019).
    """

    def __init__(
        self,
        model_key: str,
        mentions_per_tokens: float,
        antecedents_nb: int,
        max_span_size: int,
    ) -> None:
        """
        .. note::

            In the future, only ``model_key`` shall be necessary, since
            config should be read directly from the model.


        :param model_key:
        :param mentions_per_tokens:
        :param antecedents_nb:
        :param max_span_size:
        """
        self.bert_for_corefs = BertForCoreferenceResolution.from_pretrained(
            model_key, mentions_per_tokens, antecedents_nb, max_span_size
        )  # type: ignore
        self.bert_for_corefs = cast(BertForCoreferenceResolution, self.bert_for_corefs)
        # TODO: param
        self.bert_for_corefs = self.bert_for_corefs.to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )  # type: ignore

        # TODO: tokenizer key
        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")  # type: ignore

        super().__init__()

    def __call__(self, text: str, tokens: List[str], **kwargs) -> Dict[str, Any]:
        # TODO: param
        block_size = 128
        blocks = [
            tokens[block_start : block_start + block_size]
            for block_start in range(0, len(tokens), block_size)
        ]
        # TODO: hardcoded batch size
        coref_docs = self.bert_for_corefs.predict(blocks, self.tokenizer, 4)
        return {"corefs": [doc.coref_chains for doc in coref_docs]}

    def needs(self) -> Set[str]:
        return {"tokens"}

    def production(self) -> Set[str]:
        return {"corefs"}
