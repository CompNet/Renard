from __future__ import annotations
from typing import List, Dict, Any, Set, Tuple, Optional, Union, Literal
from dataclasses import dataclass
import torch
from transformers.tokenization_utils_base import BatchEncoding
from transformers import PreTrainedModel
from seqeval.metrics import precision_score, recall_score, f1_score
from renard.nltk_utils import nltk_fix_bio_tags
from renard.pipeline.core import PipelineStep, Mention
from renard.pipeline.progress import ProgressReporter


@dataclass
class NEREntity(Mention):
    #: NER class (without BIO prefix as in ``PER`` and not ``B-PER``)
    tag: str

    def shifted(self, shift: int) -> NEREntity:
        """
        .. note::

            This method is implemtented here to avoid type issues.  Since
            :meth:`.Mention.shifted` cannot be annotated as returning
            ``Self``, this method annotate the correct return type when
            using :meth:`.NEREntity.shifted`.
        """
        return super().shifted(shift)  # type: ignore


def ner_entities(
    tokens: List[str], bio_tags: List[str], resolve_inconsistencies: bool = True
) -> List[NEREntity]:
    """Extract NER entities from a list of BIO tags

    :param tokens: a list of tokens
    :param bio_tags: a list of BIO tags.  In particular, BIO tags
        should be in the CoNLL-2002 form (such as 'B-PER I-PER')

    :return: A list of ner entities, in apparition order
    """
    assert len(tokens) == len(bio_tags)

    entities = []
    current_tag: Optional[str] = None
    current_tag_start_idx: Optional[int] = None

    for i, tag in enumerate(bio_tags):
        if not current_tag is None and not tag.startswith("I-"):
            assert not current_tag_start_idx is None
            entities.append(
                NEREntity(
                    tokens[current_tag_start_idx:i],
                    current_tag_start_idx,
                    i,
                    current_tag,
                )
            )
            current_tag = None
            current_tag_start_idx = None

        if tag.startswith("B-"):
            current_tag = tag[2:]
            current_tag_start_idx = i

        elif tag.startswith("I-"):
            if current_tag is None and resolve_inconsistencies:
                current_tag = tag[2:]
                current_tag_start_idx = i
                continue

    if not current_tag is None:
        assert not current_tag_start_idx is None
        entities.append(
            NEREntity(
                tokens[current_tag_start_idx : len(tokens)],
                current_tag_start_idx,
                len(bio_tags),
                current_tag,
            )
        )

    return entities


def score_ner(
    pred_bio_tags: List[str], ref_bio_tags: List[str]
) -> Tuple[float, float, float]:
    """Score NER as in CoNLL-2003 shared task using ``seqeval``

    Precision is the percentage of named entities in ``ref_bio_tags``
    that are correct. Recall is the percentage of named entities in
    pred_bio_tags that are in ref_bio_tags. F1 is the harmonic mean of
    both.

    :param pred_bio_tags:
    :param ref_bio_tags:
    :return: ``(precision, recall, F1 score)``

    """
    assert len(pred_bio_tags) == len(ref_bio_tags)
    return (
        precision_score([ref_bio_tags], [pred_bio_tags]),
        recall_score([ref_bio_tags], [pred_bio_tags]),
        f1_score([ref_bio_tags], [pred_bio_tags]),
    )


class NLTKNamedEntityRecognizer(PipelineStep):
    """An entity recognizer based on NLTK"""

    def __init__(self) -> None:
        """
        :param language: iso 639-2 3-letter language code
        """
        import nltk

        nltk.download("averaged_perceptron_tagger", quiet=True)
        nltk.download("maxent_ne_chunker", quiet=True)
        nltk.download("words", quiet=True)

        super().__init__()

    def __call__(self, tokens: List[str], **kwargs) -> Dict[str, Any]:
        """
        :param text:
        :param tokens:
        """
        import nltk
        from nltk.chunk import tree2conlltags

        word_tag_iobtags = tree2conlltags(
            nltk.ne_chunk(nltk.pos_tag(tokens, lang=self.lang))
        )
        bio_tags = nltk_fix_bio_tags([wti[2] for wti in word_tag_iobtags])
        return {"entities": ner_entities(tokens, bio_tags)}

    def supported_langs(self) -> Union[Set[str], Literal["any"]]:
        # POS Tagging only supports english and russian
        return {"eng", "rus"}

    def needs(self) -> Set[str]:
        return {"tokens"}

    def production(self) -> Set[str]:
        return {"entities"}


class BertNamedEntityRecognizer(PipelineStep):
    """An entity recognizer based on BERT"""

    LANG_TO_MODELS = {
        "fra": "Davlan/bert-base-multilingual-cased-ner-hrl",
        "eng": "compnet-renard/bert-base-cased-literary-NER",
    }

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, str]] = None,
        batch_size: int = 4,
        device: Literal["cpu", "cuda", "auto"] = "auto",
    ):
        """
        :param huggingface_model_id: a custom huggingface model id.
            This allows to bypass the ``lang`` pipeline parameter
            which normally choose a huggingface model automatically.
        :param batch_size: batch size at inference
        :param device: computation device
        """
        if isinstance(model, str):
            self.huggingface_model_id = model
            self.model = None  # model will be init by _pipeline_init_
        else:
            self.huggingface_model_id = None
            self.model = model

        self.batch_size = batch_size

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        super().__init__()

    def _pipeline_init_(self, lang: str, progress_reporter: ProgressReporter):
        from transformers import AutoModelForTokenClassification, AutoTokenizer  # type: ignore

        super()._pipeline_init_(lang, progress_reporter)

        # init model if needed (this happens if the user did not pass
        # the instance of a model)
        if self.model is None:
            # the user supplied a huggingface ID: load model from the HUB
            if not self.huggingface_model_id is None:
                self.model = AutoModelForTokenClassification.from_pretrained(
                    self.huggingface_model_id
                )
                self.lang = "unknown"  # we don't know the lang of the custom model
            # the user did not supply anything: load the default model
            else:
                model_str = BertNamedEntityRecognizer.LANG_TO_MODELS.get(lang)
                if model_str is None:
                    raise ValueError(
                        f"BertNamedEntityRecognizer does not support language {lang}"
                    )
                self.model = AutoModelForTokenClassification.from_pretrained(model_str)
                self.tokenizer = AutoTokenizer.from_pretrained(model_str)

    def __call__(
        self,
        tokens: List[str],
        sentences: List[List[str]],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        :param text:
        :param tokens:
        :param sentences:
        """
        assert not self.model is None

        import torch

        self.model = self.model.to(self.device)

        batchs = self.encode(sentences)

        # TODO: iteration could be done using a torch dataloader
        #       instead of doing it by hand
        batches_nb = len(batchs["input_ids"]) // self.batch_size + 1

        with torch.no_grad():
            wp_labels = []

            for batch_i in self._progress_(range(batches_nb)):
                batch_start = batch_i * self.batch_size
                batch_end = batch_start + self.batch_size
                out = self.model(
                    batchs["input_ids"][batch_start:batch_end].to(self.device),
                    attention_mask=batchs["attention_mask"][batch_start:batch_end].to(
                        self.device
                    ),
                )
                # (batch_size, sentence_size)
                batch_classes_tens = torch.max(out.logits, dim=2).indices
                wp_labels += [
                    self.model.config.id2label[tens.item()]
                    for classes_tens in batch_classes_tens
                    for tens in classes_tens
                ]

            labels = self.batch_labels(batchs, 0, wp_labels, tokens)

        return {"entities": ner_entities(tokens, labels)}

    def encode(self, sentences: List[List[str]]) -> BatchEncoding:
        return self.tokenizer(
            sentences,
            return_tensors="pt",
            padding=True,
            truncation=True,
            is_split_into_words=True,
        )

    def batch_labels(
        self,
        batchs: BatchEncoding,
        batch_i: int,
        wp_labels: List[str],
        tokens: List[str],
    ) -> List[str]:
        """Align labels to tokens rather than wordpiece tokens.

        :param batchs: huggingface batch
        :param batch_i: batch index
        :param wp_labels: wordpiece aligned labels
        :param tokens: original tokens
        """
        batch_labels = ["O"] * len(tokens)

        for wplabel_j, wp_label in enumerate(wp_labels):
            token_i = batchs.token_to_word(batch_i, wplabel_j)
            if token_i is None:
                continue
            batch_labels[token_i] = wp_label

        return batch_labels

    def supported_langs(self) -> Union[Set[str], Literal["any"]]:
        return {"eng", "fra"}

    def needs(self) -> Set[str]:
        return {"tokens", "sentences"}

    def production(self) -> Set[str]:
        return {"entities"}
