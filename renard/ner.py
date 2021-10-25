from typing import List, Dict, Any, Set, Tuple, Optional
from transformers.tokenization_utils_base import BatchEncoding
from tqdm import tqdm
from renard.pipeline import PipelineStep


def bio_entities(tokens: List[str], bio_tags: List[str]) -> List[Tuple[str, str, int]]:
    """
    :return: ``(full entity string, tag, token index)``
    """
    assert len(tokens) == len(bio_tags)

    bio_entities = []

    current_entity: Optional[str] = None
    current_tag: Optional[str] = None

    for i, (token, tag) in enumerate(zip(tokens, bio_tags)):

        if tag.startswith("B-"):
            current_entity = token
            current_tag = tag[2:]

        elif tag.startswith("I-"):
            if current_entity is None:
                print(f"[warning] inconsistant bio tags. Will try to procede")
                current_entity = token
                current_tag = tag[2:]
                continue
            current_entity += f" {token}"

        elif not current_entity is None:
            bio_entities.append((current_entity, current_tag, i))
            current_entity = None
            current_tag = None

    return bio_entities


class NLTKNamedEntityRecognizer(PipelineStep):
    """An entity recognizer based on NLTK

    requires the following pre-computed pipeline attributes :

    - ``tokens: List[str]``

    :ivar language: iso 639-2 3-letter language code
    """

    def __init__(self, language: str = "eng") -> None:
        self.language = language

    def __call__(self, text: str, tokens: List[str], **kwargs) -> Dict[str, Any]:

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

    def produces(self) -> Set[str]:
        return {"bio_tags"}


class BertNamedEntityRecognizer(PipelineStep):
    """An entity recognizer based on BERT"""

    def __init__(
        self, huggingface_model_id: str = "dslim/bert-base-NER", batch_size: int = 4
    ):
        from transformers import AutoModelForTokenClassification

        self.model = AutoModelForTokenClassification.from_pretrained(
            huggingface_model_id
        )
        self.batch_size = batch_size

    def __call__(
        self,
        text: str,
        bert_batch_encoding: BatchEncoding,
        wp_tokens: List[str],
        **kwargs,
    ) -> Dict[str, Any]:
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

    def produces(self) -> Set[str]:
        return {"wp_bio_tags", "bio_tags"}
