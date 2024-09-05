from __future__ import annotations
from typing import List, Literal, Optional, Set, Dict, Any, Union, TYPE_CHECKING
from importlib import import_module
import torch
from more_itertools import windowed
from renard.pipeline import PipelineStep, Mention, ProgressReporter

if TYPE_CHECKING:
    from tibert import BertForCoreferenceResolution
    from transformers import PreTrainedTokenizerFast


class BertCoreferenceResolver(PipelineStep):
    """
    A coreference resolver using BERT.  Loosely based on 'End-to-end
    Neural Coreference Resolution' (Lee et at.  2017) and 'BERT for
    coreference resolution' (Joshi et al.  2019).
    """

    def __init__(
        self,
        model: Optional[Union[BertForCoreferenceResolution]] = None,
        huggingface_model_id: Optional[str] = None,
        batch_size: int = 1,
        device: Literal["auto", "cuda", "cpu"] = "auto",
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
        block_size: int = 512,
        hierarchical_merging: bool = False,
    ) -> None:
        """
        .. note::

            In the future, only ``mentions_per_tokens``,
            ``antecedents_nb`` and ``max_span_size`` shall be read
            directly from the model's config.

        :param huggingface_model_id: a custom huggingface model id.
            This allows to bypass the ``lang`` pipeline parameter
            which normally choose a huggingface model automatically.
        :param batch_size: batch size at inference
        :param device: computation device
        :param block_size: size of blocks to pass to the coreference
            model
        :param hierarchical_merging: if ``True``, attempts to use
            tibert's hierarchical merging feature.  In that case,
            blocks of size ``block_size`` are merged to perform
            inference on the whole document.
        """
        if isinstance(model, str):
            self.hugginface_model_id = huggingface_model_id
            self.model = None  # model will be init by _pipeline_init_
        else:
            self.hugginface_model_id = None
            self.model = model

        self.tokenizer = tokenizer

        self.batch_size = batch_size

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.block_size = block_size
        self.hierarchical_merging = hierarchical_merging

        super().__init__()

    def _pipeline_init_(self, lang: str, **kwargs):
        from tibert import BertForCoreferenceResolution
        from transformers import BertTokenizerFast, AutoTokenizer

        if self.model is None:
            # the user supplied a huggingface ID: load model from the HUB
            if not self.hugginface_model_id is None:
                self.model = BertForCoreferenceResolution.from_pretrained(
                    self.hugginface_model_id
                )
                if self.tokenizer is None:
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        self.hugginface_model_id
                    )
                self.lang = "unknown"

            # the user did not supply anything: load the default model
            else:
                self.model = BertForCoreferenceResolution.from_pretrained(
                    "compnet-renard/bert-base-cased-literary-coref"
                )
                self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

        assert not self.tokenizer is None

        super()._pipeline_init_(lang, **kwargs)

    def __call__(self, tokens: List[str], **kwargs) -> Dict[str, Any]:
        from tibert import stream_predict_coref, predict_coref
        from tibert.bertcoref import CoreferenceDocument

        blocks = [
            tokens[block_start : block_start + self.block_size]
            for block_start in range(0, len(tokens), self.block_size)
        ]

        if self.hierarchical_merging:
            doc = predict_coref(
                blocks,
                self.model,
                self.tokenizer,
                batch_size=self.batch_size,
                quiet=True,
                device_str=self.device,
                hierarchical_merging=True,
            )
            return {"corefs": doc.coref_chains}

        coref_docs = []
        for doc in self._progress_(
            stream_predict_coref(
                blocks,
                self.model,
                self.tokenizer,
                batch_size=self.batch_size,
                quiet=True,
                device_str=self.device,
            ),
            total=len(blocks),
        ):
            coref_docs.append(doc)

        return {"corefs": CoreferenceDocument.concatenated(coref_docs).coref_chains}

    def needs(self) -> Set[str]:
        return {"tokens"}

    def production(self) -> Set[str]:
        return {"corefs"}

    def supported_langs(self) -> Union[Set[str], Literal["any"]]:
        return {"eng"}


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
        dynamic_blocks_tokens: Optional[List[List[str]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        from spacy.tokens import Doc
        from coreferee.manager import CorefereeBroker

        if dynamic_blocks_tokens is None:
            dynamic_blocks_tokens = [tokens]

        if len(dynamic_blocks_tokens) > 1:
            chunks = []
            for block in dynamic_blocks_tokens:
                chunks += self._cut_into_chunks(block)
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
                        mention_tokens[-1].i + chunk_start + 1,
                    )
                    cur_chain.append(mention)

                if len(cur_chain) > 0:
                    chains.append(cur_chain)

            chunk_start += len(chunk_tokens)

        return {"corefs": chains}

    def needs(self) -> Set[str]:
        return {"tokens"}

    def optional_needs(self) -> Set[str]:
        return {"dynamic_blocks_tokens"}

    def production(self) -> Set[str]:
        return {"corefs"}
