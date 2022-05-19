from __future__ import annotations
from typing import Dict, List, Literal, Optional, Tuple, Union
import re
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from transformers import BertPreTrainedModel
from transformers import PreTrainedTokenizerFast
from transformers.file_utils import PaddingStrategy
from transformers.data.data_collator import DataCollatorMixin
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.configuration_bert import BertConfig
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from tqdm import tqdm
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

        # spans in a coref chain : mark all antecedents
        for chain in self.coref_chains:
            for mention in chain:
                try:
                    mention_idx = spans_idx.index((mention.start_idx, mention.end_idx))
                    for other_mention in chain:
                        if other_mention == mention:
                            continue
                        other_mention_idx = spans_idx.index(
                            (other_mention.start_idx, other_mention.end_idx)
                        )
                        labels[mention_idx][other_mention_idx] = 1
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
        """Construct a CoreferenceDocument using labels

        :param tokens:
        :param labels: ``(spans_nb, spans_nb + 1)``
        :param max_span_size:
        """
        spans_idx = spans_indexs(tokens, max_span_size)

        chains = []
        already_visited_mentions = []
        for i, mention_labels in enumerate(labels):
            if mention_labels[-1] == 1:
                continue
            if i in already_visited_mentions:
                continue
            start_idx, end_idx = spans_idx[i]
            mention_tokens = tokens[start_idx : end_idx + 1]
            chain = [CoreferenceMention(start_idx, end_idx, tokens=mention_tokens)]
            for j, label in enumerate(mention_labels):
                if label == 0:
                    continue
                start_idx, end_idx = spans_idx[j]
                mention_tokens = tokens[start_idx : end_idx + 1]
                chain.append(
                    CoreferenceMention(start_idx, end_idx, tokens=mention_tokens)
                )
                already_visited_mentions.append(j)
            chains.append(chain)

        return CoreferenceDocument(tokens, chains)


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

    # (batch_size, top_mentions_nb, antecedents_nb + 1)
    logits: torch.Tensor

    # (batch_size, top_mentions_nb)
    top_mentions_index: torch.Tensor

    # (batch_size, top_mentions_nb, antecedents_nb)
    top_antecedents_index: torch.Tensor

    max_span_size: int

    loss: Optional[torch.Tensor] = None

    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None

    def coreference_documents(
        self, tokens: List[List[str]]
    ) -> List[CoreferenceDocument]:
        """"""

        batch_size = self.logits.shape[0]
        top_mentions_nb = self.logits.shape[1]
        antecedents_nb = self.logits.shape[2]

        _, antecedents_idx = torch.max(self.logits, 2)
        assert antecedents_idx.shape == (batch_size, top_mentions_nb)

        documents = []

        for i in range(batch_size):

            spans_idx = spans_indexs(tokens[i], self.max_span_size)

            document = CoreferenceDocument(tokens[i], [])

            for j in range(top_mentions_nb):

                span_idx = int(self.top_mentions_index[i][j].item())
                span_coords = spans_idx[span_idx]

                top_antecedent_idx = int(antecedents_idx[i][j].item())

                # the antecedent is the dummy mention : skip
                if top_antecedent_idx == antecedents_nb - 1:
                    continue

                antecedent_idx = int(
                    self.top_antecedents_index[i][j][top_antecedent_idx].item()
                )

                antecedent_coords = spans_idx[antecedent_idx]

                # antecedent
                chain_id = None
                for chain_i, chain in enumerate(document.coref_chains):
                    for mention in chain:
                        if antecedent_coords == (mention.start_idx, mention.end_idx):
                            chain_id = chain_i
                if chain_id is None:
                    # new chain
                    chain_id = len(document.coref_chains)
                    # create a new chain in the document
                    start_idx, end_idx = antecedent_coords[0], antecedent_coords[1]
                    mention = CoreferenceMention(
                        start_idx, end_idx, tokens[i][start_idx : end_idx + 1]
                    )
                    document.coref_chains.append([mention])

                # current span
                start_idx, end_idx = span_coords[0], span_coords[1]
                mention = CoreferenceMention(
                    start_idx, end_idx, tokens[i][start_idx : end_idx + 1]
                )
                document.coref_chains[chain_id].append(mention)

            documents.append(document)

        return documents


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
        mentions_per_tokens: float,
        antecedents_nb: int,
        max_span_size: int,
        **kwargs,
    ):
        super().__init__(config, **kwargs)

        self.bert = BertModel(config, add_pooling_layer=False)

        self.mentions_per_tokens = mentions_per_tokens
        self.antecedents_nb = antecedents_nb
        self.max_span_size = max_span_size

        self.mention_scorer_hidden = torch.nn.Linear(2 * config.hidden_size, 150)
        self.mention_scorer = torch.nn.Linear(150, 1)

        self.mention_compatibility_scorer_hidden = torch.nn.Linear(
            4 * config.hidden_size, 150
        )
        self.mention_compatibility_scorer = torch.nn.Linear(150, 1)

        self.loss_fn = torch.nn.CrossEntropyLoss()

    def mention_score(self, span_bounds: torch.Tensor) -> torch.Tensor:
        """Compute a score representing how likely it is that a span is a mention

        :param span_bounds: a tensor of shape ``(batch_size, 2, hidden_size)``,
            representing the first and last token of a span.

        :return: a tensor of shape ``(batch_size)``.
        """
        # (batch, 150)
        score = self.mention_scorer_hidden(torch.flatten(span_bounds, 1))
        score = torch.relu(score)
        # (batch)
        return self.mention_scorer(score).squeeze(-1)
        # (batch_size)
        # return self.mention_scorer(torch.flatten(span_bounds, 1)).squeeze(-1)

    def mention_compatibility_score(self, span_bounds: torch.Tensor) -> torch.Tensor:
        """
        :param span_bounds: ``(batch_size, 4 * hidden_size)``

        :return: a tensor of shape ``(batch_size)``
        """
        # (batch_size, 150)
        score = self.mention_compatibility_scorer_hidden(span_bounds)
        score = torch.relu(score)
        return self.mention_compatibility_scorer(score).squeeze(-1)
        # return self.mention_compatibility_scorer(span_bounds).squeeze(-1)

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
        :param top_mentions_nb: the maximum number of spans to keep
            during the pruning process

        :return: a tensor of shape ``(batch_size, <= top_mentions_nb)``
        """
        batch_size = mention_scores.shape[0]
        spans_nb = mention_scores.shape[1]
        device = next(self.parameters()).device

        assert top_mentions_nb <= spans_nb

        spans_idx = spans_indexs(list(range(seq_size)), self.max_span_size)

        def spans_are_overlapping(
            span1: Tuple[int, int], span2: Tuple[int, int]
        ) -> bool:
            return (
                span1[0] < span2[0] and span2[0] <= span1[1] and span1[1] < span2[1]
            ) or (span2[0] < span1[0] and span1[0] <= span2[1] and span2[1] < span1[1])

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

        # To construct a tensor, we need all lists of mention to be
        # the same size : to do so, we cut them to have the length of
        # the smallest one
        min_top_mentions_nb = min([len(m) for m in mention_indexs])
        mention_indexs = [m[:min_top_mentions_nb] for m in mention_indexs]

        mention_indexs = torch.tensor(mention_indexs, device=device)

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
        :param pred_scores: ``(batch_size, top_mentions_nb, antecedents_nb + 1)``
        :param labels: ``(batch_size, top_mentions_nb, antecedents_nb + 1)``
        :return: ``(batch_size)``
        """
        # (batch_size, top_mentions_nb, antecedents_nb + 1)
        coreference_log_probs = torch.log_softmax(pred_scores, dim=-1)
        # (batch_size, top_mentions_nb, antecedents_nb + 1)
        correct_antecedent_log_probs = coreference_log_probs + labels.log()
        # (1)
        return -torch.logsumexp(correct_antecedent_log_probs, dim=-1).sum(dim=1)

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

        # top_mentions_index is the index of the m best
        # non-overlapping mentions
        top_mentions_nb = m = int(self.mentions_per_tokens * seq_size)
        top_mentions_index = self.pruned_mentions_indexs(
            mention_scores, seq_size, top_mentions_nb
        )
        # TODO: hack
        top_mentions_nb = m = int(top_mentions_index.shape[1])
        assert top_mentions_index.shape == (b, m)

        # antecedents_index contains the index of the a closest
        # antecedents for each spans
        antecedents_nb = a = min(self.antecedents_nb, top_mentions_nb)
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

        top_antecedent_scores = batch_index_select(
            mention_scores, 1, top_antecedents_index.flatten(start_dim=1)
        ).reshape(b, m, a)

        top_partial_scores = top_mention_scores.unsqueeze(-1) + top_antecedent_scores
        assert top_partial_scores.shape == (b, m, a)

        dummy = torch.zeros(b, m, 1, device=device)
        top_partial_scores = torch.cat([top_partial_scores, dummy], dim=2)
        assert top_partial_scores.shape == (b, m, a + 1)

        final_scores = top_partial_scores + mention_pair_scores
        assert final_scores.shape == (b, m, a + 1)

        # -- loss computation --
        loss = None
        if labels is not None:

            selected_labels = batch_index_select(labels, 1, top_mentions_index)
            assert selected_labels.shape == (b, m, p + 1)

            selected_labels = batch_index_select(
                selected_labels.flatten(start_dim=0, end_dim=1),
                1,
                top_antecedents_index,
            ).reshape(b, m, a)

            # mentions with no antecedents are assumed to have the dummy antecedent
            dummy_labels = (1 - selected_labels).prod(-1, keepdim=True)

            selected_labels = torch.cat((selected_labels, dummy_labels), dim=2)
            assert selected_labels.shape == (b, m, a + 1)

            loss = self.loss(final_scores, selected_labels)

        return BertCoreferenceResolutionOutput(
            final_scores,
            top_mentions_index,
            top_antecedents_index,
            self.max_span_size,
            loss=loss,
            hidden_states=bert_output.hidden_states,
            attentions=bert_output.attentions,
        )

    def predict(
        self,
        documents: List[List[str]],
        tokenizer: PreTrainedTokenizerFast,
        batch_size: int,
    ) -> List[CoreferenceDocument]:
        """Predict coreference chains for a list of documents.

        .. warning::

            WIP: the returned documents tokens are wordpieces and not the original tokens.

        :param documents: A list of tokenized documents.
        :param tokenizer:
        :param batch_size:

        :return: a list of ``CoreferenceDocument``, with annotated
                 coreference chains.
        """

        dataset = CoreferenceDataset(
            [CoreferenceDocument(doc, []) for doc in documents],
            tokenizer,
            self.max_span_size,
        )
        data_collator = DataCollatorForSpanClassification(tokenizer, self.max_span_size)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, collate_fn=data_collator, shuffle=False
        )

        self = self.eval()

        preds = []

        with torch.no_grad():

            for batch in tqdm(dataloader):

                out: BertCoreferenceResolutionOutput = self(**batch)
                preds += out.coreference_documents(
                    [
                        [tokenizer.decode(t) for t in input_ids]
                        for input_ids in batch["input_ids"]
                    ]
                )

        return preds
