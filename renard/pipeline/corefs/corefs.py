from __future__ import annotations
from typing import List, Literal, Optional, Set, Dict, Any, cast
from importlib import import_module
import torch
from transformers import BertTokenizerFast  # type: ignore
from more_itertools import windowed
from renard.pipeline import PipelineStep, Mention, ProgressReporter
from renard.pipeline.corefs.bert_corefs import BertForCoreferenceResolution


class BertCoreferenceResolver(PipelineStep):
    """
    A coreference resolver using BERT.  Loosely based on 'End-to-end
    Neural Coreference Resolution' (Lee et at.  2017) and 'BERT for
    coreference resolution' (Joshi et al.  2019).

    .. warning::

        this is a work in progress and should not be used.

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


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import spacy
    from spacy.tokens import Doc
    from spacy.tokens.token import Token
    from coreferee.manager import CorefereeBroker
    from coreferee.data_model import Mention as CorefereeMention


class SpacyCorefereeCoreferenceResolver(PipelineStep):
    """A coreference resolver using spacy's corefree.

    .. note::

        - This step requires to install Renard's extra 'spacy'
        - While this step automatically install the needed spacy models, it still needs
          a manual installation of the coreferee model: ``python -m coreferee install en``
    """

    def __init__(self, max_chunk_size: Optional[int] = 10000):
        """
        :param chunk_size: coreference chunk size, in tokens
        """
        self.max_chunk_size = max_chunk_size

    def _pipeline_init_(self, lang: str, progress_reporter: ProgressReporter):
        # NOTE: spacy_transformers import is needed to load
        # "en_core_web_trf"
        import spacy_transformers

        SpacyCorefereeCoreferenceResolver._spacy_try_load_model("en_core_web_lg")
        self.nlp = SpacyCorefereeCoreferenceResolver._spacy_try_load_model(
            "en_core_web_trf"
        )
        self.nlp.remove_pipe("ner")
        super()._pipeline_init_(lang, progress_reporter)

    @staticmethod
    def _spacy_try_load_model(name: str) -> spacy.Language:  # type: ignore
        from spacy.cli import download  # type: ignore

        try:
            mod = import_module(name)
        except ModuleNotFoundError:
            download(name)
            mod = import_module(name)
        return mod.load()

    @staticmethod
    def _spacy_try_infer_spaces(tokens: List[str]) -> List[bool]:
        """
        Try to infer, for each token, if there is a space between this
        token and the next.
        """
        spaces = []
        for _, t2 in windowed(tokens, 2):
            spaces.append(not t2 in [".", "!", "?", ","])
        spaces.append(False)  # last token has no subsequent space
        return spaces

    @staticmethod
    def _coreferee_get_mention_tokens(
        coref_model: CorefereeBroker, mention_heads: CorefereeMention, doc: Doc
    ) -> List[Token]:
        """Coreferee only return mention heads for mention, and not
        the whole span.  This hack (defined in coreferee README at the
        end of part 2
        https://github.com/richardpaulhudson/coreferee#2-interacting-with-the-data-model)
        gets the whole span as a list of spacy tokens.
        """
        rules_analyzer = coref_model.annotator.rules_analyzer
        tokens = []
        for head_i in mention_heads:
            mention_tokens = rules_analyzer.get_propn_subtree(doc[head_i])
            # in the case of non-noun mention, the previous function
            # returns an empty list. In that case, we simply return
            # the sole token at index head_i.
            if len(mention_tokens) == 0:
                tokens.append(doc[head_i])
                continue
            tokens += mention_tokens
        return tokens

    def _cut_into_chunks(self, tokens: List[str]) -> List[List[str]]:
        if self.max_chunk_size is None:
            return [tokens]
        chunks = []
        for chunk_start in range(0, len(tokens), self.max_chunk_size):
            chunk_end = chunk_start + self.max_chunk_size
            chunks.append(tokens[chunk_start:chunk_end])
        return chunks

    def __call__(
        self,
        text: str,
        tokens: List[str],
        chapter_tokens: Optional[List[List[str]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        from spacy.tokens import Doc
        from coreferee.manager import CorefereeBroker

        if chapter_tokens is None:
            chapter_tokens = [tokens]

        if len(chapter_tokens) > 1:
            chunks = []
            for chapter in chapter_tokens:
                chunks += self._cut_into_chunks(chapter)
        else:
            chunks = self._cut_into_chunks(tokens)

        chains = []

        chunk_start = 0
        for chunk_tokens in chunks:

            # see https://spacy.io/api/doc for how to instantiate a spacy doc
            spaces = SpacyCorefereeCoreferenceResolver._spacy_try_infer_spaces(
                chunk_tokens
            )
            spacy_doc = Doc(self.nlp.vocab, words=chunk_tokens, spaces=spaces)

            # current steps in the spacy pipeline:
            # - transformer
            # - tagger
            # - parser
            # - attribute_ruler
            # - lemmatization
            for _, step in self.nlp.pipeline:
                spacy_doc = step(spacy_doc)

            coref_model = CorefereeBroker(self.nlp, "coref_chains")
            spacy_doc = coref_model(spacy_doc)

            # * parse coreferee chains
            for chain in spacy_doc._.coref_chains:

                cur_chain = []

                for mention in chain:

                    mention_tokens = (
                        SpacyCorefereeCoreferenceResolver._coreferee_get_mention_tokens(
                            coref_model, mention, spacy_doc
                        )
                    )

                    # some spans produced by coreferee are not contigous:
                    # chains containing such spans are considered
                    # *invalid* for Renard, and are discarded
                    span_is_contiguous = len(mention_tokens) == 1 or all(
                        [t1.i == t2.i - 1 for t1, t2 in windowed(mention_tokens, 2)]
                    )

                    if not span_is_contiguous:
                        cur_chain = []
                        break

                    mention = Mention(
                        [str(t) for t in mention_tokens],
                        mention_tokens[0].i + chunk_start,
                        mention_tokens[-1].i + chunk_start,
                    )
                    cur_chain.append(mention)

                if len(cur_chain) > 0:
                    chains.append(cur_chain)

            chunk_start += len(chunk_tokens)

        return {"corefs": chains}

    def needs(self) -> Set[str]:
        return {"tokens"}

    def optional_needs(self) -> Set[str]:
        return {"chapter_tokens"}

    def production(self) -> Set[str]:
        return {"corefs"}
