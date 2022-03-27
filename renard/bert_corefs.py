from __future__ import annotations
from typing import Dict, List, Literal, Optional, Tuple, Union, cast
import re
from dataclasses import dataclass
from more_itertools import windowed
import torch
from torch.functional import tensordot
from torch.utils.data import Dataset
from transformers import BertPreTrainedModel
from transformers import PreTrainedTokenizerFast
from transformers.file_utils import PaddingStrategy
from transformers.data.data_collator import DataCollatorMixin
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.configuration_bert import BertConfig
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from renard.utils import spans, spans_indexs, batch_index_select


@dataclass
class CoreferenceMention:
    start_idx: int
    end_idx: int
    tokens: List[str]


@dataclass
class CoreferenceDocument:
    tokens: List[str]
    coref_chains: List[List[CoreferenceMention]]

    def __len__(self) -> int:
        return len(self.tokens)

    def document_labels(self, max_span_size: int) -> List[List[int]]:
        """
        :return: a list of shape ``(spans_nb, spans_nb + 1)``.
            when ``out[i][j] == 1``, span j is the preceding
            coreferent mention if span i. when ``j == spans_nb``,
            i has no preceding coreferent mention.
        """
        spans_idx = spans_indexs(self.tokens, max_span_size)
        spans_nb = len(spans_idx)

        # labels = torch.zeros(spans_nb, spans_nb + 1)
        labels = [[0] * (spans_nb + 1) for _ in range(spans_nb)]

        # spans with a preceding mention : mark the preceding mention
        for chain in self.coref_chains:
            for prev_mention, mention in windowed(chain, 2):
                mention = cast(CoreferenceMention, mention)
                prev_mention = cast(CoreferenceMention, prev_mention)
                try:
                    mention_idx = spans_idx.index((mention.start_idx, mention.end_idx))
                    prev_mention_idx = spans_idx.index(
                        (prev_mention.start_idx, prev_mention.end_idx)
                    )
                    labels[mention_idx][prev_mention_idx] = 1
                except ValueError:
                    continue

        # spans without preceding mentions : mark preceding mention to
        # be the null span
        for i in range(len(labels)):
            if all(l == 0 for l in labels[i]):
                labels[i][spans_nb] = 1

        return labels

    def retokenized_document(
        self, tokenizer: PreTrainedTokenizerFast
    ) -> CoreferenceDocument:
        """Returns a new document, retokenized thanks to ``tokenizer``.
        In particular, coreference chains are adapted to the newly
        tokenized text.

        .. note::

            The passed tokenizer is called using its ``__call__``
            method. This means special tokens will be added.

        :param tokenizer: tokenizer used to retokenized the document
        :return: a new :class:`CoreferenceDocument`
        """
        # (silly) exemple for the tokens ["I", "am", "PG"]
        # a BertTokenizer would produce ["[CLS]", "I", "am", "P", "##G", "[SEP]"]
        batch = tokenizer(self.tokens, is_split_into_words=True)  # type: ignore
        tokens = tokenizer.convert_ids_to_tokens(batch["input_ids"])

        # words_ids is used to correspond post-tokenization word pieces
        # to their corresponding pre-tokenization tokens.
        # for our above example, word_ids would then be : [None, 0, 1, 2, 2, None]
        words_ids = batch.word_ids(batch_index=0)
        # reversed words ids will be used to compute mention end index
        # in the post-tokenization sentence later
        r_words_ids = list(reversed(words_ids))

        new_chains = []
        for chain in self.coref_chains:
            new_chains.append([])
            for mention in chain:
                # compute [start_index, end_index] of the mention in
                # the retokenized sentence
                # start_idx is the index of the first word-piece corresponding
                # to the word at its original start index.
                start_idx = words_ids.index(mention.start_idx)
                # end_idx is the index of the last word-piece corresponding
                # to the word at its original end index.
                end_idx = len(words_ids) - 1 - r_words_ids.index(mention.end_idx)
                new_chains[-1].append(
                    CoreferenceMention(
                        start_idx, end_idx, tokens[start_idx : end_idx + 1]
                    )
                )

        return CoreferenceDocument(tokens, new_chains)

    @staticmethod
    def from_labels(
        tokens: List[str], labels: List[List[int]], max_span_size: int
    ) -> CoreferenceDocument:
        """
        :param tokens:
        :param labels:
        :param max_span_size:
        """
        spans_idx = spans_indexs(tokens, max_span_size)

        # last known chain mention index => chain
        chains: Dict[int, List[CoreferenceMention]] = {}
        for i, labels_line in enumerate(labels):
            if labels_line[-1] == 1:
                # span has no antecedent : nothing to do
                continue
            antecedent_index = labels_line.index(1)
            if not antecedent_index in chains:
                # span has an antecedent and this antecedent is the first
                # mention of a chain : create a new chain
                start_idx, end_idx = spans_idx[antecedent_index]
                chains[antecedent_index] = [
                    CoreferenceMention(
                        start_idx, end_idx, tokens[start_idx : end_idx + 1]
                    )
                ]
            # add current span to its chain
            start_idx, end_idx = spans_idx[i]
            chains[antecedent_index].append(
                CoreferenceMention(start_idx, end_idx, tokens[start_idx : end_idx + 1])
            )
            # set last known antecedent index to current index
            chains[i] = chains.pop(antecedent_index)

        return CoreferenceDocument(tokens, list(chains.values()))


@dataclass
class DataCollatorForSpanClassification(DataCollatorMixin):
    """
    .. note::

        Only implements the torch data collator.
    """

    tokenizer: PreTrainedTokenizerBase
    max_span_size: int
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: Literal["pt"] = "pt"

    def torch_call(self, features) -> Union[dict, BatchEncoding]:
        labels = (
            [feature["labels"] for feature in features]
            if "labels" in features[0].keys()
            else None
        )
        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        # keep encoding info
        batch._encodings = [f.encodings[0] for f in features]

        if labels is None:
            return batch

        documents = [
            CoreferenceDocument.from_labels(
                tokens, labels, max_span_size=self.max_span_size
            )
            for tokens, labels in zip(
                [f["input_ids"] for f in features], [f["labels"] for f in features]
            )
        ]

        for document, tokens in zip(documents, batch["input_ids"]):  # type: ignore
            document.tokens = tokens
        batch["labels"] = [
            document.document_labels(self.max_span_size) for document in documents
        ]

        return BatchEncoding(
            {k: torch.tensor(v, dtype=torch.int64) for k, v in batch.items()},
            encoding=batch.encodings,
        )


class CoreferenceDataset(Dataset):
    """
    :ivar documents:
    :ivar tokenizer:
    :ivar max_span_len:
    """

    def __init__(
        self,
        documents: List[CoreferenceDocument],
        tokenizer: PreTrainedTokenizerFast,
        max_span_size: int,
    ) -> None:
        super().__init__()
        self.documents = documents
        self.tokenizer = tokenizer
        self.max_span_size = max_span_size

    def __len__(self) -> int:
        return len(self.documents)

    def __getitem__(self, index: int) -> BatchEncoding:
        document = self.documents[index]
        document = document.retokenized_document(self.tokenizer)
        batch = self.tokenizer(
            document.tokens, is_split_into_words=True, add_special_tokens=False
        )  # type: ignore
        batch["labels"] = document.document_labels(self.max_span_size)
        return batch


class WikiCorefDataset(CoreferenceDataset):
    """The WikiCoref dataset (http://rali.iro.umontreal.ca/rali/?q=en/wikicoref)"""

    def __init__(
        self, path: str, tokenizer: PreTrainedTokenizerFast, max_span_size: int
    ) -> None:
        """
        :param path: path to the root of the WikiCoref dataset
            downloaded from http://rali.iro.umontreal.ca/rali/?q=en/wikicoref
        """
        path = path.rstrip("/")

        documents = []
        document_tokens = []
        # dict chain id => coref chain
        document_chains: Dict[str, List[CoreferenceMention]] = {}
        # dict chain id => list of mention start index
        open_mentions: Dict[str, List[int]] = {}

        with open(f"{path}/Evaluation/key-OntoNotesScheme") as f:

            for line in f:

                line = line.rstrip("\n")

                if line.startswith("null") or re.fullmatch(r"\W*", line):
                    continue

                if line.startswith("#end document"):
                    document = CoreferenceDocument(
                        document_tokens, list(document_chains.values())
                    )
                    documents.append(document)
                    continue

                if line.startswith("#begin document"):
                    document_tokens = []
                    document_chains = {}
                    open_mentions = {}
                    continue

                splitted = line.split("\t")

                # - tokens
                document_tokens.append(splitted[3])

                # - coreference datas parsing
                #
                # coreference datas are indicated as follows in the dataset. Either :
                #
                # * there is a single dash ("-"), indicating no datas
                #
                # * there is an ensemble of coref datas, separated by pipes ("|") if
                #   there are more than 2. coref datas are of the form "(?[0-9]+)?"
                #   (example : "(71", "(71)", "71)").
                #   - A starting parenthesis indicate the start of a mention
                #   - A ending parenthesis indicate the end of a mention
                #   - The middle number indicates the ID of the coreference chain
                #     the mention belongs to
                if splitted[4] == "-":
                    continue

                coref_datas_list = splitted[4].split("|")
                for coref_datas in coref_datas_list:

                    mention_is_starting = coref_datas.find("(") != -1
                    mention_is_ending = coref_datas.find(")") != -1
                    chain_id = re.search(r"[0-9]+", coref_datas).group(0)

                    if mention_is_starting:
                        open_mentions[chain_id] = open_mentions.get(chain_id, []) + [
                            len(document_tokens) - 1
                        ]

                    if mention_is_ending:
                        mention_start_idx = open_mentions[chain_id].pop()
                        mention_end_idx = len(document_tokens) - 1
                        mention = CoreferenceMention(
                            mention_start_idx,
                            mention_end_idx,
                            document_tokens[mention_start_idx : mention_end_idx + 1],
                        )
                        document_chains[chain_id] = document_chains.get(
                            chain_id, []
                        ) + [mention]

        super().__init__(documents, tokenizer, max_span_size)


@dataclass
class BertCoreferenceResolutionOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class BertForCoreferenceResolution(BertPreTrainedModel):
    """BERT for Coreference Resolution

    .. note ::

        We use the following short notation to annotate shapes :

        - b : batch_size
        - s : seq_size
        - p : spans_nb
        - m : top_mentions_nb
        - a : antecedents_nb
        - h : hidden_size
    """

    def __init__(
        self,
        config: BertConfig,
        top_mentions_nb: int,
        antecedents_nb: int,
        max_span_size: int,
    ):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)

        self.top_mentions_nb = top_mentions_nb
        self.antecedents_nb = antecedents_nb
        self.max_span_size = max_span_size

        self.mention_scorer = torch.nn.Linear(2 * config.hidden_size, 1)
        self.mention_compatibility_scorer = torch.nn.Linear(4 * config.hidden_size, 1)

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def mention_score(self, span_bounds: torch.Tensor) -> torch.Tensor:
        """Compute a score representing how likely it is that a span is a mention

        :param span_bounds: a tensor of shape ``(batch_size, 2, hidden_size)``,
            representing the first and last token of a span.

        :return: a tensor of shape ``(batch_size)``.
        """
        # (batch_size)
        return self.mention_scorer(torch.flatten(span_bounds, 1)).squeeze(-1)

    def mention_compatibility_score(self, span_bounds: torch.Tensor) -> torch.Tensor:
        """
        :param span_bounds: ``(batch_size, 4 * hidden_size)``

        :return: a tensor of shape ``(batch_size)``
        """
        return self.mention_compatibility_scorer(span_bounds).squeeze(-1)

    def pruned_mentions_indexs(
        self, mention_scores: torch.Tensor, seq_size: int, top_mentions_nb: int
    ) -> torch.Tensor:
        """Prune mentions, keeping only the k non-overlapping best of them

        The algorithm works as follows :

        1. Sort mentions by individual scores
        2. Accept mention in orders, from best to worst score, until k of
            them are accepted. A mention can only be accepted if no
            previously accepted span os overlapping with it.

        See section 5 of the E2ECoref paper and the C++ kernel in the
        E2ECoref repository.


        :param mention_scores: a tensor of shape ``(batch_size, spans_nb)``
        :param seq_size:
        :param top_mentions_nb: the maximum number of spans to keep during the pruning process
        :return: a tensor of shape ``(batch_size, <= k)``
        """
        batch_size = mention_scores.shape[0]
        spans_nb = mention_scores.shape[1]
        device = next(self.parameters()).device

        spans_idx = spans_indexs(list(range(seq_size)), self.max_span_size)

        def spans_are_overlapping(
            span1: Tuple[int, int], span2: Tuple[int, int]
        ) -> bool:
            return (span1[1] <= span2[1] and span1[1] >= span2[0]) or (
                span1[0] >= span2[0] and span1[0] <= span2[1]
            )

        _, sorted_indexs = torch.sort(mention_scores, 1, descending=True)
        # TODO: what if we can't have top_mentions_nb mentions ??
        mention_indexs = []
        # TODO: optim
        for i in range(batch_size):
            mention_indexs.append([])
            for j in range(spans_nb):
                if len(mention_indexs[-1]) >= top_mentions_nb:
                    break
                span_index = int(sorted_indexs[i][j].item())
                if not any(
                    [
                        spans_are_overlapping(
                            spans_idx[span_index], spans_idx[mention_idx]
                        )
                        for mention_idx in mention_indexs[-1]
                    ]
                ):
                    mention_indexs[-1].append(sorted_indexs[i][j])
        mention_indexs = torch.tensor(mention_indexs, device=device)
        assert mention_indexs.shape == (batch_size, top_mentions_nb)

        return mention_indexs

    def closest_antecedents_indexs(
        self, spans_nb: int, seq_size: int, antecedents_nb: int
    ) -> torch.Tensor:
        """Compute the indexs of the k closest mentions

        TODO: optim

        :param spans_nb:
        :param seq_size:
        :param antecedents_nb:
        :return: a tensor of shape ``(spans_nb, antecedents_nb)``
        """
        device = next(self.parameters()).device

        def antecedent_dist(
            span: Tuple[int, int], antecedent: Tuple[int, int]
        ) -> float:
            if antecedent[1] >= span[0]:
                return float("Inf")
            return span[0] - antecedent[1]

        spans_idx = spans_indexs(list(range(seq_size)), self.max_span_size)
        dist_matrix = torch.zeros(spans_nb, spans_nb, device=device)
        for i in range(spans_nb):
            for j in range(spans_nb):
                dist_matrix[i][j] = antecedent_dist(spans_idx[i], spans_idx[j])

        _, close_indexs = torch.topk(-dist_matrix, antecedents_nb)
        assert close_indexs.shape == (spans_nb, antecedents_nb)

        return close_indexs

    def loss(self, pred_scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        :param pred_scores: ``(batch_size, spans_nb, spans_nb + 1)``
        :param labels: ``(batch_size, spans_nb, spans_nb + 1)``
        :return: ``(batch_size)``
        """
        return self.loss_fn(pred_scores.transpose(1, 2), labels.transpose(1, 2).float())

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        TODO: add return type

        :param input_ids: a tensor of shape ``(batch_size, seq_size)``
        :param attention_mask: a tensor of shape ``(batch_size, seq_size)``
        :param labels: a tensor of shape ``(batch_size, spans_nb, spans_nb)``
        """

        batch_size = b = input_ids.shape[0]
        seq_size = s = input_ids.shape[1]
        hidden_size = h = self.config.hidden_size

        device = next(self.parameters()).device

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        bert_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        encoded_input = bert_output.last_hidden_state
        assert encoded_input.shape == (b, s, h)

        # -- span bounds computation --
        # we select starting and ending bounds of spans of length up
        # to self.max_span_size
        spans_idx = spans(range(seq_size), self.max_span_size)
        spans_nb = p = len(spans_idx)
        spans_selector = torch.flatten(
            torch.tensor([[span[0], span[-1]] for span in spans_idx], dtype=torch.long)
        ).to(device)
        assert spans_selector.shape == (p * 2,)
        span_bounds = torch.index_select(encoded_input, 1, spans_selector)
        span_bounds = span_bounds.reshape(b, p, 2, h)

        # -- mention scores computation --
        mention_scores = self.mention_score(
            torch.flatten(span_bounds, start_dim=0, end_dim=1)
        )
        assert mention_scores.shape == (b * p,)
        mention_scores = mention_scores.reshape(b, p)

        # -- pruning thanks to mention scores --
        # TODO: top_mentions_nb should be a parameter
        #       depending on the length of input sentences
        #       (it is denoted as 'lambda * T' in the paper)
        top_mentions_nb = m = 3

        # top_mentions_index is the index of the m best
        # non-overlapping mentions
        top_mentions_index = self.pruned_mentions_indexs(
            mention_scores, seq_size, top_mentions_nb
        )
        assert top_mentions_index.shape == (b, m)

        # antecedents_index contains the index of the a closest
        # antecedents for each spans
        antecedents_nb = a = min(3, top_mentions_nb)
        antecedents_index = self.closest_antecedents_indexs(
            spans_nb, seq_size, antecedents_nb
        )
        antecedents_index = torch.tile(antecedents_index, (batch_size, 1, 1))
        assert antecedents_index.shape == (b, p, a)

        # -- mention compatibility scores computation --
        # top_mentions_bounds keep only span bounds for spans with enough score
        top_mentions_bounds = batch_index_select(span_bounds, 1, top_mentions_index)
        assert top_mentions_bounds.shape == (b, m, 2, h)

        top_antecedents_index = batch_index_select(
            antecedents_index, 1, top_mentions_index
        )
        assert top_antecedents_index.shape == (b, m, a)

        top_antecedents_bounds = batch_index_select(
            span_bounds, 1, top_antecedents_index.flatten(start_dim=1)
        )
        top_antecedents_bounds = top_antecedents_bounds.reshape(b, m, a, 2, h)

        # span_bounds_combination is a tensor containing the
        # representation of each possible pair of mentions. Each
        # representation is of shape (4, hidden_size). the first
        # dimension (4) represents the number of tokens used in a pair
        # representation (first token of first span, last token of
        # first span, first token of second span and last token of
        # second span). There are spans_nb ^ 2 such representations.
        #
        # /!\ below code could be optimised and has WIP status
        #     see https://github.com/mandarjoshi90/coref/blob/master/overlap.py
        #     for inspiration.
        span_bounds_combination = torch.stack(
            [
                # each tensor has shape : (batch_size, 4, hidden_size)
                torch.cat(
                    [
                        # (batch_size, 2, hidden_size)
                        top_mentions_bounds[:, i, :, :],
                        # (batch_size, 2, hidden_size)
                        top_antecedents_bounds[:, i, j, :, :],
                    ],
                    1,
                )
                for i in range(top_mentions_nb)
                for j in range(antecedents_nb)
            ],
            dim=1,
        )
        assert span_bounds_combination.shape == (b, m * a, 4, h)
        span_bounds_combination = torch.flatten(span_bounds_combination, start_dim=2)
        assert span_bounds_combination.shape == (b, m * a, 4 * h)

        mention_pair_scores = self.mention_compatibility_score(
            torch.flatten(span_bounds_combination, start_dim=0, end_dim=1)
        )
        assert mention_pair_scores.shape == (b * m * a,)
        mention_pair_scores = mention_pair_scores.reshape(b, m, a)

        # add in dummy mention scores
        dummy_scores = torch.zeros(batch_size, top_mentions_nb, 1, device=device)
        mention_pair_scores = torch.cat(
            [
                mention_pair_scores,
                dummy_scores,
            ],
            dim=2,
        )
        assert mention_pair_scores.shape == (b, m, a + 1)

        # -- final mention scores computation --
        top_mention_scores = torch.gather(mention_scores, 1, top_mentions_index)
        assert top_mention_scores.shape == (b, m)

        # s_m(m1)
        # TODO: use torch.tile
        mention1_score = torch.cat(
            [
                torch.stack(
                    [top_mention_scores for _ in range(top_mentions_nb)], dim=1
                ),
                torch.zeros(batch_size, top_mentions_nb, 1, device=device),
            ],
            dim=2,
        )
        assert mention1_score.shape == (b, m, a + 1)

        # s_m(m2)
        # TODO: use torch.tile
        mention2_score = torch.stack(
            [top_mention_scores for _ in range(antecedents_nb + 1)], dim=1
        ).transpose(1, 2)
        assert mention2_score.shape == (b, m, a + 1)

        # s_m(m1) + s_m(m2) + s_c(m1, m2)
        final_scores = mention1_score + mention2_score + mention_pair_scores
        assert final_scores.shape == (b, m, a + 1)

        # reconstruct scores
        # TODO: perf
        # TODO: correctness
        full_final_scores = torch.full(
            (b, p, p + 1), -100, device=device, dtype=torch.float
        )
        for b in range(batch_size):
            for i in range(top_mentions_nb):
                m_idx = int(top_mentions_index[b][i].item())
                for j in range(antecedents_nb):  # TODO: + 1
                    a_idx = int(antecedents_index[b][i][j].item())
                    full_final_scores[b][m_idx][a_idx] = final_scores[b][i][j]
                full_final_scores[b][m_idx][-1] = final_scores[b][i][-1]

        # -- loss computation --
        loss = None
        if labels is not None:
            loss = self.loss(full_final_scores, labels)

        return BertCoreferenceResolutionOutput(
            logits=final_scores,
            loss=loss,
            hidden_states=bert_output.hidden_states,
            attentions=bert_output.attentions,
        )
