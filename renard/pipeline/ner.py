from typing import List, Dict, Any, Set, Tuple, Optional
from dataclasses import dataclass
from transformers.tokenization_utils_base import BatchEncoding
from tqdm import tqdm
from seqeval.metrics import precision_score, recall_score, f1_score
from renard.pipeline.core import PipelineStep


@dataclass
class NEREntity:
    #: entitiy tokens
    tokens: List[str]
    #: NER class (without BIO prefix as in ``PER`` and not ``B-PER``)
    tag: str
    #: entity start index
    start_idx: int
    #: entity end index
    end_idx: int


def ner_entities(
    tokens: List[str], bio_tags: List[str], resolve_inconsistencies: bool = True
) -> List[NEREntity]:
    """Extract NER entities from a list of BIO tags

    :param tokens:
    :param bio_tags:
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
                    current_tag,
                    current_tag_start_idx,
                    i - 1,
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
                current_tag,
                current_tag_start_idx,
                len(bio_tags) - 1,
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

    def __init__(self, language: str = "eng") -> None:
        """
        :param language: iso 639-2 3-letter language code
        """
        self.language = language
        super().__init__()

    def __call__(self, text: str, tokens: List[str], **kwargs) -> Dict[str, Any]:
        """
        :param text:
        :param tokens:
        """

        import nltk
        from nltk.chunk import tree2conlltags

        nltk.download("averaged_perceptron_tagger", quiet=True)
        nltk.download("maxent_ne_chunker", quiet=True)
        nltk.download("words", quiet=True)

        word_tag_iobtags = tree2conlltags(
            nltk.ne_chunk(nltk.pos_tag(tokens, lang=self.language))
        )
        return {"bio_tags": [wti[2] for wti in word_tag_iobtags]}

    def needs(self) -> Set[str]:
        return {"tokens"}

    def production(self) -> Set[str]:
        return {"bio_tags"}


class BertNamedEntityRecognizer(PipelineStep):
    """An entity recognizer based on BERT"""

    def __init__(
        self,
        model: str = "dslim/bert-base-NER",
        batch_size: int = 4,
    ):
        """
        :param model: huggingface model id or path to a custom model.
            Is passed to the huggingface ``from_pretrained`` method.
        :param batch_size:
        """
        from transformers import AutoModelForTokenClassification

        self.model = AutoModelForTokenClassification.from_pretrained(model)
        self.batch_size = batch_size
        super().__init__()

    def __call__(
        self,
        text: str,
        bert_batch_encoding: BatchEncoding,
        wp_tokens: List[str],
        **kwargs,
    ) -> Dict[str, Any]:
        """
        :param text:
        :param bert_batch_encoding:
        :param wp_tokens:
        """
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

        batches_nb = len(bert_batch_encoding["input_ids"]) // self.batch_size + 1

        with torch.no_grad():
            wp_labels = []

            batch_i_s = (
                tqdm(range(batches_nb))
                if self.progress_report == "tqdm"
                else range(batches_nb)
            )
            for batch_i in batch_i_s:
                batch_start = batch_i * self.batch_size
                batch_end = batch_start + self.batch_size
                out = self.model(
                    bert_batch_encoding["input_ids"][batch_start:batch_end].to(device),
                    attention_mask=bert_batch_encoding["attention_mask"][
                        batch_start:batch_end
                    ].to(device),
                )
                # (batch_size, sentence_size)
                batch_classes_tens = torch.max(out.logits, dim=2).indices
                wp_labels += [
                    self.model.config.id2label[tens.item()]
                    for classes_tens in batch_classes_tens
                    for tens in classes_tens
                ]
            labels = BertNamedEntityRecognizer.wp_labels_to_token_labels(
                wp_tokens, wp_labels
            )

        return {
            "wp_bio_tags": wp_labels,
            "bio_tags": labels,
        }

    @staticmethod
    def wp_labels_to_token_labels(
        wp_tokens: List[str], wp_labels: List[str]
    ) -> List[str]:
        """Output a list of labels aligned with regular tokens instead
        of word piece tokens.

        :param wp_tokens: word piece tokens
        :param wp_labels: word piece labels
        """
        assert len(wp_tokens) == len(wp_labels)
        labels = []
        for (wp_token, wp_label) in zip(wp_tokens, wp_labels):
            if wp_token in {"[CLS]", "[SEP]", "[PAD]"}:
                continue
            if not wp_token.startswith("##"):
                labels.append(wp_label)
        return labels

    def needs(self) -> Set[str]:
        return {"bert_batch_encoding", "wp_tokens"}

    def production(self) -> Set[str]:
        return {"wp_bio_tags", "bio_tags"}
