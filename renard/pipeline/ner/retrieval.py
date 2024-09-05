from collections.abc import Set
import sys
from typing import Union, List, cast, Literal, Optional
import random
from dataclasses import dataclass
from more_itertools import flatten
from renard.ner_utils import NERDataset
import nltk
from rank_bm25 import BM25Okapi
from transformers import (
    BertForSequenceClassification,
    BertTokenizerFast,
    DataCollatorWithPadding,
)
from transformers.tokenization_utils_base import BatchEncoding
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class NERContextRetrievalMatch:
    element: List[str]
    element_i: int
    side: Literal["left", "right"]
    score: Optional[float]

    def __hash__(self) -> int:
        return hash(tuple(self.element) + (self.element_i, self.side, self.score))


class NERContextRetriever:
    def __init__(self, k: int) -> None:
        self.k = k

    def compute_global_features(self, elements: List[List[str]]) -> dict:
        return {}

    def retrieve(
        self, element_i: int, elements: List[List[str]], **kwargs
    ) -> List[NERContextRetrievalMatch]:
        raise NotImplementedError

    def __call__(self, dataset: NERDataset) -> NERDataset:
        # [(left_ctx, element, right_ctx), ...]
        elements_with_context = []

        global_features = self.compute_global_features(dataset.elements)

        for elt_i, elt in enumerate(dataset.elements):
            matchs = self.retrieve(elt_i, dataset.elements, **global_features)
            assert len(matchs) <= self.k

            lctx = sorted(
                (m for m in matchs if m.side == "left"),
                key=lambda m: m.element_i,
            )
            lctx = list(flatten([m.element for m in lctx]))

            rctx = sorted(
                (m for m in matchs if m.side == "right"),
                key=lambda m: m.element_i,
            )
            rctx = list(flatten([m.element for m in rctx]))

            elements_with_context.append((lctx, elt, rctx))

        return NERDataset(
            [lctx + element + rctx for lctx, element, rctx in elements_with_context],
            dataset.tokenizer,
            [
                [1] * len(lctx) + [0] * len(element) + [1] * len(rctx)
                for lctx, element, rctx in elements_with_context
            ],
        )


class NERSamenounContextRetriever(NERContextRetriever):
    """
    Retrieve relevant context using the samenoun strategy as in
    Amalvy et al.  2023.
    """

    def __init__(self, k: int) -> None:
        """
        :param k: the max number of sentences to retrieve
        """
        super().__init__(k)

    def compute_global_features(self, elements: List[List[str]]) -> dict:
        return {
            "NNs": [
                {t[0] for t in nltk.pos_tag(element) if t[1] == "NN"}
                for element in elements
            ]
        }

    def retrieve(
        self, element_i: int, elements: List[List[str]], NNs: List[Set[str]], **kwargs
    ) -> List[NERContextRetrievalMatch]:
        matchs = [
            NERContextRetrievalMatch(
                other_elt,
                other_elt_i,
                "left" if other_elt_i < element_i else "right",
                None,
            )
            for other_elt_i, other_elt in enumerate(elements)
            if not other_elt_i == element_i
            and len(NNs[element_i].intersection(NNs[other_elt_i])) > 0  # type: ignore
        ]
        return random.sample(matchs, k=min(self.k, len(matchs)))


class NERNeighborsContextRetriever(NERContextRetriever):
    """A context retriever that chooses nearby elements."""

    def __init__(self, k: int):
        assert k % 2 == 0
        super().__init__(k)

    def retrieve(
        self, element_i: int, elements: List[List[str]], **kwargs
    ) -> List[NERContextRetrievalMatch]:
        left_nb = self.k // 2
        right_nb = left_nb

        lctx = []
        for i, elt in enumerate(elements[element_i - left_nb : element_i]):
            lctx.append(
                NERContextRetrievalMatch(elt, element_i - left_nb + i, "left", None)
            )

        rctx = []
        for i, elt in enumerate(elements[element_i + 1 : element_i + 1 + right_nb]):
            rctx.append(NERContextRetrievalMatch(elt, element_i + 1 + i, "right", None))

        return lctx + rctx


class NERBM25ContextRetriever(NERContextRetriever):
    """A context retriever that selects elements according to the BM25 ranking formula."""

    def __init__(self, k: int) -> None:
        super().__init__(k)

    def compute_global_features(self, elements: List[List[str]]) -> dict:
        return {"bm25_model": BM25Okapi(elements)}

    def retrieve(
        self, element_i: int, elements: List[List[str]], bm25_model: BM25Okapi, **kwargs
    ) -> List[NERContextRetrievalMatch]:
        query = elements[element_i]
        sent_scores = bm25_model.get_scores(query)
        sent_scores[element_i] = float("-Inf")  # don't retrieve self
        topk_values, topk_indexs = torch.topk(
            torch.tensor(sent_scores), k=min(self.k, len(sent_scores)), dim=0
        )
        return [
            NERContextRetrievalMatch(
                elements[index], index, "left" if index < element_i else "right", value
            )
            for value, index in zip(topk_values.tolist(), topk_indexs.tolist())
        ]


@dataclass(frozen=True)
class NERNeuralContextRetrievalExample:
    """A context retrieval example."""

    #: text on which NER is performed
    element: List[str]
    #: context to assist during prediction
    context: List[str]
    #: context side (does the context comes from the left or the right of ``sent`` ?)
    context_side: Literal["left", "right"]


class NERNeuralContextRetrievalDataset(Dataset):
    """"""

    def __init__(
        self,
        examples: List[NERNeuralContextRetrievalExample],
        tokenizer: BertTokenizerFast,
    ) -> None:
        self.examples = examples
        self.tokenizer: BertTokenizerFast = tokenizer

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> BatchEncoding:
        """Get a BatchEncoding representing example at index.

        :param index: index of the example to retrieve

        :return: a ``BatchEncoding``, with key ``'label'`` set.
        """
        example = self.examples[index]

        tokens = example.context + ["[SEP]"] + example.element

        batch: BatchEncoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            truncation=True,
            max_length=512,
        )

        return batch


class NERNeuralContextRetriever(NERContextRetriever):
    """
    A neural context retriever as in Amalvy et al.  2024
    """

    def __init__(
        self,
        heuristic_context_selector: NERContextRetriever,
        pretrained_model: Union[
            str, BertForSequenceClassification
        ] = "compnet-renard/bert-base-cased-NER-reranker",
        k: int = 3,
        batch_size: int = 1,
        threshold: float = 0.0,
        device_str: Literal["cuda", "cpu", "auto"] = "auto",
    ) -> None:
        """
        :param pretrained_model: pretrained model name, used to
            load a :class:`transformers.BertForSequenceClassification`
        :param heuristic_context_selector: name of the context
            selector to use as retrieval heuristic, from
            ``context_selector_name_to_class``
        :param heuristic_context_selector_kwargs: kwargs to pass the
            heuristic context retriever at instantiation time
        :param k: max number of sents to retrieve
        :param batch_size: batch size used at inference
        :param threshold:
        :param device_str:
        """
        from transformers import BertForSequenceClassification, BertTokenizerFast

        if isinstance(pretrained_model, str):
            self.ctx_classifier = BertForSequenceClassification.from_pretrained(
                pretrained_model
            )  # type: ignore
        else:
            self.ctx_classifier = pretrained_model
        self.ctx_classifier = cast(BertForSequenceClassification, self.ctx_classifier)

        self.tokenizer = BertTokenizerFast.from_pretrained(
            pretrained_model if isinstance(pretrained_model, str) else "bert-base-cased"
        )

        self.heuristic_context_selector = heuristic_context_selector

        self.batch_size = batch_size
        self.threshold = threshold

        if device_str == "auto":
            device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)

        super().__init__(k)

    def set_heuristic_k_(self, k: int):
        self.heuristic_context_selector.k = k

    def predict(self, examples: List[NERNeuralContextRetrievalExample]) -> torch.Tensor:
        """
        :param dataset: A list of :class:`ContextSelectionExample`
        :return: A tensor of shape ``(len(dataset), 2)`` of class
                 scores
        """
        dataset = NERNeuralContextRetrievalDataset(examples, self.tokenizer)

        self.ctx_classifier = self.ctx_classifier.to(self.device)

        data_collator = DataCollatorWithPadding(dataset.tokenizer)  # type: ignore
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, collate_fn=data_collator)  # type: ignore

        # inference using self.ctx_classifier
        self.ctx_classifier = self.ctx_classifier.eval()
        with torch.no_grad():
            scores = torch.zeros((0,)).to(self.device)
            for X in dataloader:
                X = X.to(self.device)
                # out.logits is of shape (batch_size, 2)
                out = self.ctx_classifier(
                    X["input_ids"],
                    token_type_ids=X["token_type_ids"],
                    attention_mask=X["attention_mask"],
                )
                # (batch_size, 2)
                pred = torch.softmax(out.logits, dim=1)
                scores = torch.cat([scores, pred], dim=0)

        return scores

    def compute_global_features(self, elements: List[List[str]]) -> dict:
        features = self.heuristic_context_selector.compute_global_features(elements)
        return {
            "heuristic_matchs": [
                self.heuristic_context_selector.retrieve(i, elements, **features)
                for i in range(len(elements))
            ]
        }

    def retrieve(
        self,
        element_i: int,
        elements: List[List[str]],
        heuristic_matchs: List[List[NERContextRetrievalMatch]],
        **kwargs,
    ) -> List[NERContextRetrievalMatch]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ctx_classifier = self.ctx_classifier.to(device)  # type: ignore

        # no context retrieved by heuristic : nothing to do
        if len(heuristic_matchs) == 0:
            return []

        element = elements[element_i]
        matchs = heuristic_matchs[element_i]

        # prepare datas for inference
        ctx_dataset = [
            NERNeuralContextRetrievalExample(element, m.element, m.side) for m in matchs
        ]

        # (len(dataset), 2)
        scores = self.predict(ctx_dataset)
        for i, m in enumerate(matchs):
            m.score = float(scores[i, 1].item())

        assert all([not m.score is None for m in matchs])
        return [
            m
            for m in sorted(matchs, key=lambda m: -m.score)[: self.k]  # type: ignore
            if m.score > self.threshold  # type: ignore
        ]


class NEREnsembleContextRetriever(NERContextRetriever):
    """Combine several context retriever"""

    def __init__(self, retrievers: List[NERContextRetriever], k: int) -> None:
        self.retrievers = retrievers
        super().__init__(k)

    def compute_global_features(self, elements: List[List[str]]) -> dict:
        features = {}
        for retriever in self.retrievers:
            for k, v in retriever.compute_global_features(elements).items():
                if k in features:
                    print(
                        f"[warning] NEREnsembleContextRetriver: incompatible global feature '{k}' between multiple retrievers.",
                        file=sys.stderr,
                    )
                features[k] = v
        return features

    def retrieve(
        self, element_i: int, elements: List[List[str]], **kwargs
    ) -> List[NERContextRetrievalMatch]:
        all_matchs = set()

        for retriever in self.retrievers:
            matchs = retriever.retrieve(element_i, elements, **kwargs)
            all_matchs = all_matchs.union(matchs)

        if all(not m.score is None for m in all_matchs):
            return sorted(all_matchs, key=lambda m: -m.score)[: self.k]  # type: ignore
        return random.choices(list(all_matchs), k=self.k)
