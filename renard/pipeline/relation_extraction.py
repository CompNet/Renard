from typing import Any, Union, Optional, Literal
import ast, re
import functools as ft
from datasets import load_dataset, Dataset as HGDataset
import torch
from transformers import (
    AutoModelForSeq2SeqLM,
    T5ForConditionalGeneration,
    AutoTokenizer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    PreTrainedTokenizerFast,
    DataCollatorForSeq2Seq,
    PreTrainedModel,
    pipeline as hg_pipeline,
)
from transformers.pipelines.pt_utils import KeyDataset
from renard.pipeline.core import PipelineStep
from renard.pipeline.progress import ProgressReporter
from renard.pipeline.character_unification import Character

#: (subject, relation, object)
Relation = tuple[Character, str, Character]


def _load_ARF_line(example: dict, tokenizer: PreTrainedTokenizerFast) -> dict:
    example["relations"] = ast.literal_eval(example["relations"] or "[]")

    def format_rel(rel: dict) -> str:
        return "({}, {}, {})".format(rel["entity1"], rel["relation"], rel["entity2"])

    labels = " ".join(map(format_rel, example["relations"]))
    with tokenizer.as_target_tokenizer():
        labels_batch = tokenizer(labels)
    example["labels"] = labels_batch["input_ids"]

    text = example["chunk"] or ""
    text = f"extract relations: {text}"
    example["input_ids"] = tokenizer(text)["input_ids"]

    return example


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
    dataset = dataset.train_test_split(test_size=0.1)
    return dataset.map(ft.partial(_load_ARF_line, tokenizer=tokenizer))


def train_t5_on_ARF(
    t5_hg_id: str, targs: Seq2SeqTrainingArguments
) -> T5ForConditionalGeneration:
    tokenizer = AutoTokenizer.from_pretrained(t5_hg_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(t5_hg_id)

    dataset = load_ARF_dataset(tokenizer)

    trainer = Seq2SeqTrainer(
        model,
        targs,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=DataCollatorForSeq2Seq(tokenizer),
    )
    trainer.train()

    return model


class T5RelationExtractor(PipelineStep):
    DEFAULT_MODEL = "compnet-renard/t5-small-literary-relation-extraction"

    def __init__(
        self,
        model: Optional[Union[PreTrainedModel, str]] = None,
        batch_size: int = 1,
        device: Literal["cpu", "cuda", "auto"] = "auto",
    ):
        self.model = T5RelationExtractor.DEFAULT_MODEL if model is None else model
        self.hg_pipeline = None
        self.batch_size = batch_size
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

    def _pipeline_init_(self, lang: str, progress_reporter: ProgressReporter, **kwargs):
        super()._pipeline_init_(lang, progress_reporter, **kwargs)
        self.hg_pipeline = hg_pipeline(
            "text2text-generation", model=self.model, device=self.device
        )

    def __call__(
        self, sentences: list[list[str]], characters: list[Character], **kwargs
    ) -> dict[str, Any]:
        assert not self.hg_pipeline is None

        sentence_relations = []

        # chunk as in the ARF dataset
        dataset = HGDataset.from_list(
            [{"text": T5RelationExtractor.task_prompt(sent)} for sent in sentences]
        )
        for out in self._progress_(
            self.hg_pipeline(KeyDataset(dataset, "text"), batch_size=self.batch_size),
            total=len(dataset),
        ):
            text_relations = out[0]["generated_text"]

            raw_triples = T5RelationExtractor.parse_t5_text_relations(text_relations)
            triples = []
            for subj, rel, obj in raw_triples:
                subj_char = T5RelationExtractor.identify_character(subj, characters)
                obj_char = T5RelationExtractor.identify_character(obj, characters)
                if subj_char is None or obj_char is None or subj_char == obj_char:
                    continue
                triples.append((subj_char, rel, obj_char))
            sentence_relations.append(triples)

        return {"sentence_relations": sentence_relations}

    @staticmethod
    def task_prompt(sentence: list[str]) -> str:
        sent_text = " ".join(sentence)
        return f"extract relations: {sent_text}"

    @staticmethod
    def parse_t5_text_relations(text_relations: str) -> list[tuple[str, str, str]]:
        return re.findall(r"\(([^,]+), ([^,]+), ([^,]+)\)", text_relations)

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
