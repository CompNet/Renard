from typing import List, Dict, Any, Set, Tuple, Optional, Union
import torch
from transformers.tokenization_utils_base import BatchEncoding
from tqdm import tqdm
from renard.pipeline.core import PipelineStep


def bio_entities(tokens: List[str], bio_tags: List[str]) -> List[Tuple[str, str, int]]:
    """
    :return: ``(full entity string, tag, token index)``
    """
    assert len(tokens) == len(bio_tags)

    bio_entities = []

    current_entity: Optional[str] = None
    current_tag: Optional[str] = None
    current_i: Optional[int] = None

    inconsistent_tags_count = 0

    def maybe_push_current_entity():
        nonlocal current_entity, current_tag, current_i
        if current_entity is None:
            return
        bio_entities.append((current_entity, current_tag, current_i))
        current_entity = None
        current_tag = None
        current_i = None

    for i, (token, tag) in enumerate(zip(tokens, bio_tags)):
        if tag.startswith("B-"):
            maybe_push_current_entity()
            current_entity = token
            current_tag = tag[2:]
            current_i = i
        elif tag.startswith("I-"):
            if current_entity is None:
                inconsistent_tags_count += 1
                current_entity = token
                current_tag = tag[2:]
                current_i = i
                continue
            current_entity += f" {token}"
        else:
            maybe_push_current_entity()
    maybe_push_current_entity()

    if inconsistent_tags_count > 0:
        print(f"[warning] inconsistent bio tags (x{inconsistent_tags_count})")

    return bio_entities


def score_ner(
    pred_bio_tags: List[str], ref_bio_tags: List[str]
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Score NER as in CoNLL-2003 shared task

    Precision is the percentage of named entities in ``ref_bio_tags``
    that are correct. Recall is the percentage of named entities in
    pred_bio_tags that are in ref_bio_tags. F1 is the harmonic mean of
    both.

    :param pred_bio_tags:
    :param ref_bio_tags:
    :return: ``(precision, recall, F1 score)``

    """
    assert len(pred_bio_tags) == len(ref_bio_tags)

    if len(pred_bio_tags) == 0:
        return (None, None, None)

    pred_entities = []
    ref_entities = []

    for (entity_list, tags) in zip(
        [pred_entities, ref_entities], [pred_bio_tags, ref_bio_tags]
    ):

        current_entity: Optional[Dict[str, Union[int, str]]] = None

        for i, tag in enumerate(tags):

            if tag.startswith("B-"):
                if not current_entity is None:
                    current_entity["end_idx"] = i
                    entity_list.append(current_entity)
                current_entity = {"start_idx": i, "type": tag[2:]}

            elif tag.startswith("O"):
                if not current_entity is None:
                    current_entity["end_idx"] = i
                    entity_list.append(current_entity)
                current_entity = None

        if not current_entity is None:
            current_entity["end_idx"] = len(tags)
            entity_list.append(current_entity)

    # TODO: optim
    correct_predictions = 0
    for pred_entity in pred_entities:
        if pred_entity in ref_entities:
            correct_predictions += 1
    precision = None
    if len(pred_entities) > 0:
        precision = correct_predictions / len(pred_entities)

    # TODO: optim
    recalled_entities = 0
    for ref_entity in ref_entities:
        if ref_entity in pred_entities:
            recalled_entities += 1
    recall = None
    if len(ref_entities) > 0:
        recall = recalled_entities / len(ref_entities)

    if precision is None or recall is None or precision + recall == 0:
        return (precision, recall, None)
    f1 = 2 * precision * recall / (precision + recall)

    return (precision, recall, f1)


class NLTKNamedEntityRecognizer(PipelineStep):
    """An entity recognizer based on NLTK"""

    def __init__(self, language: str = "eng") -> None:
        """
        :param language: iso 639-2 3-letter language code
        """
        self.language = language

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

            for batch_i in tqdm(range(batches_nb)):
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
