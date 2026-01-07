from typing import Any, Union, Optional, Literal
import ast, re
import functools as ft
from datasets import load_dataset, Dataset as HGDataset
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerFast,
    DataCollatorForSeq2Seq,
    PreTrainedModel,
    EvalPrediction,
    pipeline as hg_pipeline,
    BatchEncoding,
)
from more_itertools import flatten
from transformers.pipelines.pt_utils import KeyDataset
from renard.pipeline.core import PipelineStep
from renard.pipeline.progress import ProgressReporter
from renard.pipeline.character_unification import Character
from renard.utils import make_vocab
from sklearn.metrics import precision_recall_fscore_support

#: (subject, relation, object)
Relation = tuple[Character, str, Character]


def _load_ARF_line(example: dict, tokenizer: PreTrainedTokenizerFast) -> BatchEncoding:
    relations = ast.literal_eval(example["relations"] or "[]")

    def format_rel(rel: dict) -> str:
        return "({}, {}, {})".format(rel["entity1"], rel["relation"], rel["entity2"])

    labels = " ".join(map(format_rel, relations))

    text = example["chunk"] or ""
    batch = tokenizer(
        tokenizer.bos_token + GenerativeRelationExtractor.task_prompt(text),
        text_target=labels + tokenizer.eos_token,
        add_special_tokens=False,
    )
    batch["relations"] = relations

    return batch


def load_ARF_dataset(tokenizer: PreTrainedTokenizerFast) -> HGDataset:
    """
    Load the Artificial Relationships in Fiction dataset
    (https://huggingface.co/datasets/Despina/project_gutenberg) by
    Christou and Tsoumakas (2025)
    """
    dataset = load_dataset(
        "Despina/project_gutenberg",
        "synthetic_relations_in_fiction_books",
        split="train",
    )
    dataset = dataset.train_test_split(test_size=0.001)
    return dataset.map(ft.partial(_load_ARF_line, tokenizer=tokenizer))


def _triple_precision_recall_f1(
    references: list[list[tuple[str, str, str]]],
    predictions: list[list[tuple[str, str, str]]],
) -> dict[str, float]:
    triple_vocab = make_vocab(list(flatten(references)) + list(flatten(predictions)))

    # the "null triple" indicates no prediction (or no reference
    # available), useful to compute precision/recall.
    null_triple_index = max(triple_vocab.values()) + 1

    y, y_hat = [], []
    for ref, pred in zip(references, predictions):
        ref = {triple: triple_vocab[triple] for triple in ref}
        pred = {triple: triple_vocab[triple] for triple in pred}
        for ref_triple, ref_index in ref.items():
            y.append(ref_index)
            y_hat.append(pred.get(ref_triple, null_triple_index))
            try:
                del pred[ref_triple]
            except KeyError:
                pass
        for pred_triple, pred_index in pred.items():
            y_hat.append(pred_index)
            y.append(ref.get(pred_triple, null_triple_index))

    precision, recall, f1, _ = precision_recall_fscore_support(
        y, y_hat, labels=list(triple_vocab.values()), average="micro"
    )

    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def train_model_on_ARF(
    model: Union[str, PreTrainedModel],
    targs: TrainingArguments,
    tokenizer: Union[PreTrainedTokenizerFast, None] = None,
) -> PreTrainedModel:
    if isinstance(model, str):
        assert tokenizer is None
        tokenizer = AutoTokenizer.from_pretrained(model)
        model = AutoModelForSeq2SeqLM.from_pretrained(model)
    assert not tokenizer is None
    tokenizer.pad_token = tokenizer.eos_token
    pad_token_i = tokenizer.encode(tokenizer.pad_token)[0]

    dataset = load_ARF_dataset(tokenizer)

    def compute_metrics(eval_preds: EvalPrediction) -> dict[str, float]:
        eval_preds.label_ids[eval_preds.label_ids == -100] = pad_token_i

        labels_str = tokenizer.batch_decode(
            eval_preds.label_ids, skip_special_tokens=True
        )
        labels = list(map(GenerativeRelationExtractor.parse_text_relations, labels_str))

        pred_ids = eval_preds.predictions[0].argmax(axis=-1)
        preds_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        preds = list(map(GenerativeRelationExtractor.parse_text_relations, preds_str))

        return _triple_precision_recall_f1(labels, preds)

    trainer = Trainer(
        model,
        targs,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    return model


class GenerativeRelationExtractor(PipelineStep):
    """

    .. warning::

        This extractor is in development and should not be used.
    """

    DEFAULT_MODEL = "compnet-renard/t5-small-literary-relation-extraction"

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, str]] = None,
        batch_size: int = 1,
        device: Literal["cpu", "cuda", "auto"] = "auto",
    ):
        self.model = (
            GenerativeRelationExtractor.DEFAULT_MODEL if model is None else model
        )
        self.hg_pipeline = None
        self.batch_size = batch_size
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def _pipeline_init_(self, lang: str, progress_reporter: ProgressReporter, **kwargs):
        super()._pipeline_init_(lang, progress_reporter, **kwargs)
        self.hg_pipeline = hg_pipeline(
            "text2text-generation",
            torch_dtype=torch.bfloat16,
            model=self.model,
            device=self.device,
        )

    def __call__(
        self, sentences: list[list[str]], characters: list[Character], **kwargs
    ) -> dict[str, Any]:
        assert not self.hg_pipeline is None

        sentence_relations = []

        # chunk as in the ARF dataset
        dataset = HGDataset.from_list(
            [
                {"text": GenerativeRelationExtractor.task_prompt(" ".join(sent))}
                for sent in sentences
            ]
        )
        for out in self._progress_(
            self.hg_pipeline(KeyDataset(dataset, "text"), batch_size=self.batch_size),
            total=len(dataset),
        ):
            text_relations = out[0]["generated_text"]

            raw_triples = GenerativeRelationExtractor.parse_text_relations(
                text_relations
            )
            triples = []
            for subj, rel, obj in raw_triples:
                subj_char = GenerativeRelationExtractor.identify_character(
                    subj, characters
                )
                obj_char = GenerativeRelationExtractor.identify_character(
                    obj, characters
                )
                if subj_char is None or obj_char is None or subj_char == obj_char:
                    continue
                triples.append((subj_char, rel, obj_char))
            sentence_relations.append(triples)

        return {"sentence_relations": sentence_relations}

    @staticmethod
    def task_prompt(text: str) -> str:
        return f"Extract triplets (subject, relation, object) from the given text: '{text}'"

    @staticmethod
    def parse_text_relations(text_relations: str) -> list[tuple[str, str, str]]:
        triplets = re.findall(
            r"\(([^,]+), ?([^,]+), ?([^,]+)\)",
            text_relations,
        )
        triplets = [
            (subj.strip(" "), rel.strip(" "), obj.strip(" "))
            for subj, rel, obj in triplets
        ]
        return triplets

    @staticmethod
    def identify_character(
        name: str, characters: list[Character]
    ) -> Optional[Character]:
        possible_character = None
        for character in characters:
            if name in character.names:
                if not possible_character is None:
                    return None
                possible_character = character
        return possible_character

    def supported_langs(self) -> set[str]:
        return {"eng"}

    def needs(self) -> set[str]:
        return {"sentences", "characters"}

    def production(self) -> set[str]:
        return {"sentence_relations"}
