from typing import List, Literal, Set, Dict, Any, cast
import torch
from transformers import BertTokenizerFast  # type: ignore
from renard.pipeline import PipelineStep, Mention
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
        device: Literal["auto", "cuda", "cpu"] = "auto",
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
        :param device: computation device
        """
        if device == "auto":
            torch_device = torch.device("cuda" if torch.cuda.is_available else "cpu")
        else:
            torch_device = torch.device(device)

        self.bert_for_corefs = BertForCoreferenceResolution.from_pretrained(
            model, mentions_per_tokens, antecedents_nb, max_span_size
        )  # type: ignore
        self.bert_for_corefs = cast(BertForCoreferenceResolution, self.bert_for_corefs)
        self.bert_for_corefs = self.bert_for_corefs.to(torch_device)  # type: ignore

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
            blocks,
            self.tokenizer,
            self.batch_size,
        )
        # chains found in coref_docs are each local to their
        # blocks. The following code adjusts their start and end index
        # to match their global coordinate in the text.
        coref_chains = []
        cur_doc_start = 0
        for doc in coref_docs:
            for chain in doc.coref_chains:
                adjusted_chain = []
                for mention in chain:
                    start_idx = mention.start_idx + cur_doc_start
                    end_idx = mention.end_idx + cur_doc_start
                    adjusted_chain.append(Mention(mention.tokens, start_idx, end_idx))
                coref_chains.append(adjusted_chain)
            cur_doc_start += len(doc)

        return {"corefs": coref_chains}

    def needs(self) -> Set[str]:
        return {"tokens"}

    def production(self) -> Set[str]:
        return {"corefs"}
