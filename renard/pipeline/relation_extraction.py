# HACK: Monkey patch transformers.tokenization_utils, because
# lmformatenforcer imports transformers.tokenization_utils which does
# not exist anymore
# fmt: off
import transformers.tokenization_utils_base
import transformers.tokenization_utils # type: ignore
transformers.tokenization_utils.PreTrainedTokenizerBase = (
    transformers.tokenization_utils_base.PreTrainedTokenizerBase
)
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import (
    build_transformers_prefix_allowed_tokens_fn,
)
# fmt: on

from typing import Any, Union, Optional, Literal, Type, Set, Tuple, cast
import ast, re, math, json, sys
from more_itertools import chunked, flatten
import functools as ft
from datasets import load_dataset, Dataset as HFDataset, DatasetDict as HFDatasetDict
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    PreTrainedTokenizerFast,
    PreTrainedModel,
    BatchEncoding,
)
from pydantic import BaseModel, create_model
from transformers.pipelines.pt_utils import KeyDataset
from renard.pipeline.core import PipelineStep, Mention
from renard.pipeline.progress import ProgressReporter
from renard.pipeline.character_unification import Character
from renard.utils import make_vocab, block_bounds
from sklearn.metrics import precision_recall_fscore_support

#: (subject, predicate, object)
Relation = tuple[Character, str, Character]

ARF_VALID_NOVELS = {
    "Blue Jackets: The Log of the Teaser",
    "Nightmare Abbey",
    "The White Chief of the Caffres",
}
ARF_TEST_NOVELS = {
    "Molly Brown's Freshman Days",
    "Ancient Rome: The Lives of Great Men",
    "The White Chief of the Caffres",
}


def _load_ARF_line(example: dict, tokenizer: PreTrainedTokenizerFast) -> BatchEncoding:
    relations = ast.literal_eval(example["relations"] or "[]")

    def format_rel(rel: dict) -> str:
        return "({}, {}, {})".format(rel["entity1"], rel["relation"], rel["entity2"])

    labels = " ".join(map(format_rel, relations))

    text = example["chunk"] or ""
    batch = tokenizer(
        tokenizer.bos_token + Seq2SeqRelationExtractor.task_prompt(text),
        text_target=labels + tokenizer.eos_token,
        add_special_tokens=False,
    )
    batch["relations"] = relations

    return batch


def load_ARF_dataset(tokenizer: PreTrainedTokenizerFast) -> HFDatasetDict:
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

    dataset = dataset.map(ft.partial(_load_ARF_line, tokenizer=tokenizer))

    ARF_TRAIN_NOVELS = set(dataset["title"]) - (ARF_VALID_NOVELS | ARF_TEST_NOVELS)
    train = dataset.filter(lambda example: example["title"] in ARF_TRAIN_NOVELS)
    valid = dataset.filter(lambda example: example["title"] in ARF_VALID_NOVELS)
    test = dataset.filter(lambda example: example["title"] in ARF_TEST_NOVELS)

    return HFDatasetDict({"train": train, "valid": valid, "test": test})  # type: ignore


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

    def compute_metrics(eval_preds) -> dict[str, float]:
        eval_preds.label_ids[eval_preds.label_ids == -100] = pad_token_i
        eval_preds.predictions[eval_preds.predictions == -100] = pad_token_i

        labels_str = tokenizer.batch_decode(
            eval_preds.label_ids, skip_special_tokens=True
        )
        labels = list(map(Seq2SeqRelationExtractor.parse_text_relations, labels_str))

        preds_str = tokenizer.batch_decode(
            eval_preds.predictions, skip_special_tokens=True
        )
        preds = list(map(Seq2SeqRelationExtractor.parse_text_relations, preds_str))

        return _triple_precision_recall_f1(labels, preds)

    trainer = Trainer(
        model,
        targs,
        train_dataset=dataset["train"],
        eval_dataset=dataset["valid"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, model),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    return model


class LLMRelationExtractor(PipelineStep):
    def __init__(
        self,
        model: Union[PreTrainedModel, str],
        relations: Set[str],
        batch_size: int = 1,
        device: Literal["cpu", "cuda", "auto"] = "auto",
        tokenizer: Optional[PreTrainedTokenizerFast] = None,
    ):
        if isinstance(model, str):
            self.huggingface_model_id = model
            self.model = None  # model will be init by _pipeline_init_
        else:
            self.huggingface_model_id = None
            self.model = model
        self.tokenizer = tokenizer

        self.batch_size = batch_size

        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.relations = relations

    def _pipeline_init_(self, lang: str, progress_reporter: ProgressReporter, **kwargs):
        super()._pipeline_init_(lang, progress_reporter, **kwargs)

        if self.model is None:
            assert not self.huggingface_model_id is None
            self.model = AutoModelForCausalLM.from_pretrained(self.huggingface_model_id)
            self.model.to(self.device)  # type: ignore
            if self.tokenizer is None:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    self.huggingface_model_id
                )
            self.lang = "unknown"  # we don't know the lang of the custom model

    def __call__(
        self, sentences: list[list[str]], characters: list[Character], **kwargs
    ) -> dict[str, Any]:  # type: ignore
        assert not self.model is None
        assert not self.tokenizer is None
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        sent_mention2character = self.sentence_mention2character(sentences, characters)
        # we only perform inference if a character mention is present in the sentence
        sent_data = [
            (i, sent, mention2char)
            for i, (sent, mention2char) in enumerate(
                zip(sentences, sent_mention2character)
            )
            if len(mention2char) > 0
        ]

        triplets: list[list[Relation]] = [[] for _ in sentences]

        for batch in self._progress_(
            chunked(sent_data, self.batch_size),
            total=math.floor(len(sent_data) / self.batch_size),
        ):
            batch_sent_i = [i for i, _, _ in batch]
            batch_sents = [sents for _, sents, _ in batch]
            batch_mention2char = [mention2char for _, _, mention2char in batch]

            batch_triplet_list_models = [
                self.make_triplet_list_model(set(mention2char.keys()))
                for mention2char in batch_mention2char
            ]
            batch_parsers = [
                JsonSchemaParser(tl.model_json_schema())
                for tl in batch_triplet_list_models
            ]
            batch_prefix_functions = [
                build_transformers_prefix_allowed_tokens_fn(self.tokenizer, parser)
                for parser in batch_parsers
            ]

            def batch_prefix_allowed_tokens_fn(
                batch_id: int, input_ids: torch.Tensor
            ) -> list[int]:
                target_fn = batch_prefix_functions[batch_id]
                return target_fn(batch_id, input_ids)

            batch_prompts = [
                self.make_task_prompt(" ".join(sent), tl)
                for sent, tl in zip(batch_sents, batch_triplet_list_models)
            ]

            inputs = self.tokenizer(
                batch_prompts, padding=True, return_tensors="pt"
            ).to(self.model.device)
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max([len(prompt) for prompt in batch_prompts]) * 4,
                prefix_allowed_tokens_fn=batch_prefix_allowed_tokens_fn,
                do_sample=False,
            )  # type: ignore

            prompt_len = inputs["input_ids"].shape[-1]
            for i in range(len(batch_prompts)):
                generated_ids = output_ids[i][prompt_len:]
                generated_text = self.tokenizer.decode(
                    generated_ids, skip_special_tokens=True
                )

                try:
                    raw_sent_triplets = json.loads(generated_text)["triplets"]  # type: ignore
                except json.decoder.JSONDecodeError as e:
                    print(
                        f"could not parse triplet from: {generated_text}",
                        file=sys.stderr,
                    )
                    print(e, file=sys.stderr)
                    raw_sent_triplets = []

                sent_triplets = []
                for triplet in raw_sent_triplets:
                    subj_char = batch_mention2char[i].get(triplet["subject"])
                    obj_char = batch_mention2char[i].get(triplet["object"])
                    if subj_char is None or obj_char is None or subj_char == obj_char:
                        continue
                    sent_triplets.append((subj_char, triplet["relation"], obj_char))

                triplets[batch_sent_i[i]] = sent_triplets

        return {"sentence_relations": triplets}

    def sentence_mention2character(
        self, sentences: list[list[str]], characters: list[Character]
    ) -> list[dict[str, Character]]:
        sent_bounds = block_bounds(sentences)[0]
        mentions = [{} for _ in range(len(sent_bounds))]

        for character in characters:
            for mention in character.mentions:
                for sent_i, (start_i, end_i) in enumerate(sent_bounds):
                    if mention.start_idx >= start_i and mention.end_idx <= end_i:
                        mentions[sent_i][" ".join(mention.tokens)] = character

        return mentions

    def make_triplet_list_model(self, mentions_str: set[str]) -> Type[BaseModel]:
        SubjectLiteral = Literal[tuple(mentions_str)]  # type: ignore
        RelationLiteral = Literal[tuple(self.relations)]  # type: ignore
        ObjectLiteral = Literal[tuple(mentions_str)]  # type: ignore
        Triplet = create_model(
            "Relation",
            subject=(SubjectLiteral, ...),
            relation=(RelationLiteral, ...),
            object=(ObjectLiteral, ...),
        )

        class TripletList(BaseModel):
            triplets: list[Triplet]  # type: ignore

        return TripletList

    def make_task_prompt(self, text: str, pydantic_model: Type[BaseModel]) -> str:
        assert not self.tokenizer is None
        schema = pydantic_model.model_json_schema()
        messages = [
            {
                "role": "system",
                "content": f"You are an expert in relation extraction. You must always answer with a JSON matching this schema: {schema}",
            },
            {
                "role": "user",
                "content": f"Extract triplets (subject, relation, object) from the given text: '{text}'\n\nRespond with a JSON object strictly matching the schema.",
            },
        ]
        # TODO: support for thinking
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )  # type: ignore

    def supported_langs(self) -> set[str]:
        return {"eng"}

    def needs(self) -> set[str]:
        return {"sentences", "characters"}

    def production(self) -> set[str]:
        return {"sentence_relations"}
