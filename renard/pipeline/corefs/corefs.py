from typing import List, Set, Dict, Any, cast
import torch
from transformers import BertTokenizerFast  # type: ignore
from renard.pipeline import PipelineStep
from renard.pipeline.corefs.bert_corefs import BertForCoreferenceResolution


class BertCoreferenceResolver(PipelineStep):
    """
    A coreference resolver using BERT.  Loosely based on 'End-to-end
    Neural Coreference Resolution' (Lee et at.  2017) and 'BERT for
    coreference resolution' (Joshi et al.  2019).
    """

    def __init__(
        self,
        model: str,
        mentions_per_tokens: float,
        antecedents_nb: int,
        max_span_size: int,
        tokenizer: str = "bert-base-cased",
        batch_size: int = 4,
        block_size: int = 128,
    ) -> None:
        """
        .. note::

            In the future, only ``mentions_per_tokens``,
            ``antecedents_nb`` and ``max_span_size`` shall be read
            directly from the model's config.

        :param model: key of the huggingface model
        :param mentions_per_tokens: number of candidate mention per
            wordpiece token
        :param antecedents_nb: max number of candidate antecedents for
            each candidate mention
        :param max_span_size: maximum size of candidate spans, in
            wordpiece tokens
        :param tokenizer: name of the hugginface tokenizer
        :param batch_size: batch size at inference
        :param block_size: size of text blocks to consider
        """
        self.bert_for_corefs = BertForCoreferenceResolution.from_pretrained(
            model, mentions_per_tokens, antecedents_nb, max_span_size
        )  # type: ignore
        self.bert_for_corefs = cast(BertForCoreferenceResolution, self.bert_for_corefs)
        # TODO: param
        self.bert_for_corefs = self.bert_for_corefs.to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )  # type: ignore

        # TODO: tokenizer key
        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer)  # type: ignore

        self.batch_size = batch_size

        self.block_size = block_size

        super().__init__()

    def __call__(self, text: str, tokens: List[str], **kwargs) -> Dict[str, Any]:
        """
        :param text:
        :param tokens:
        """
        blocks = [
            tokens[block_start : block_start + self.block_size]
            for block_start in range(0, len(tokens), self.block_size)
        ]

        coref_docs = self.bert_for_corefs.predict(
            blocks, self.tokenizer, self.batch_size
        )

        return {"corefs": [doc.coref_chains for doc in coref_docs]}

    def needs(self) -> Set[str]:
        return {"tokens"}

    def production(self) -> Set[str]:
        return {"corefs"}
