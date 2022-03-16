from typing import Optional, Tuple
from dataclasses import dataclass
import torch
from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.configuration_bert import BertConfig
from renard.utils import spans


@dataclass
class BertCoreferenceResolutionOutput:
    logits: torch.Tensor
    loss: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class BertForCoreferenceResolution(BertPreTrainedModel):
    """"""

    def __init__(self, config: BertConfig, max_span_size: int = 3):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)

        self.max_span_size = max_span_size

        self.mention_scorer = torch.nn.Linear(2 * config.hidden_size, 1)
        self.mention_compatibility_scorer = torch.nn.Linear(4 * config.hidden_size, 1)

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

        :param input_ids: a :class:`torch.Tensor` of shape ``(batch_size, seq_size)``
        :param attention_mask:
        :param labels: a :class:`torch.Tensor` of shape ``(batch_size, spans_nb, spans_nb)``
        """

        batch_size = input_ids.shape[0]
        seq_size = input_ids.shape[1]
        hidden_size = self.config.hidden_size

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
        assert encoded_input.shape == (batch_size, seq_size, hidden_size)

        # -- span bounds computation --
        # we select starting and ending bounds of spans of length up
        # to self.max_span_size
        spans_idx = spans(range(seq_size), self.max_span_size)
        spans_nb = len(spans_idx)
        spans_selector = torch.flatten(
            torch.tensor([[span[0], span[-1]] for span in spans_idx], dtype=torch.long)
        )
        assert spans_selector.shape == (spans_nb * 2,)
        span_bounds = torch.index_select(encoded_input, 1, spans_selector).reshape(
            batch_size, spans_nb, 2, hidden_size
        )
        assert span_bounds.shape == (batch_size, spans_nb, 2, hidden_size)

        # -- mention scores computation --
        mention_scores = self.mention_score(
            torch.flatten(span_bounds, start_dim=0, end_dim=1)
        )
        assert mention_scores.shape == (batch_size * spans_nb,)
        mention_scores = mention_scores.reshape((batch_size, spans_nb))
        assert mention_scores.shape == (batch_size, spans_nb)

        # -- mention compatibility scores computation --
        #
        # - TODO: pruning thanks to mention scores
        #         see =section 5= of the E2ECoref paper
        #         and the C++ kernel found in the
        #         E2ECoref repository. the algorithm
        #         seems to be as follows :
        #         1. sort mention by individual scores
        #         2. accept mentions from best score to
        #            least. A mention can only be accepted
        #            if no previously accepted span is
        #            overlapping with it. There is a limit
        #            on the number of accepted sents.

        # Q: > what is happening just below ?
        # A: > we create a tensor containing the representation
        #      of each possible pair of mentions (some
        #      mentions will be pruned for optimisation, see
        #      above). Each representation is of shape
        #      (4, hidden_size). the first dimension (4)
        #      represents the number of tokens used in a pair
        #      representation (first token of first span,
        #      last token of first span, first token of second
        #      span and last token of second span). There are
        #      spans_nb ^ 2 such representations.
        #
        # /!\ below code could be optimised and has WIP status
        #     see https://github.com/mandarjoshi90/coref/blob/master/overlap.py
        #     for inspiration.
        #
        span_bounds_combination = torch.stack(
            [
                # representation for a pair of mentions
                # (batch_size, 4, hidden_size)
                torch.flatten(
                    torch.cat((span_bounds[:, i, :, :], span_bounds[:, j, :, :]), 1),
                    start_dim=1,
                    end_dim=2,
                )
                for i in range(spans_nb)
                for j in range(spans_nb)
            ],
            dim=1,
        )
        assert span_bounds_combination.shape == (
            batch_size,
            spans_nb ** 2,
            4 * hidden_size,
        )

        mention_compat_scores = self.mention_compatibility_score(
            torch.flatten(span_bounds_combination, start_dim=0, end_dim=1)
        )
        assert mention_compat_scores.shape == (batch_size * spans_nb ** 2,)
        mention_compat_scores = mention_compat_scores.reshape(
            (batch_size, spans_nb, spans_nb)
        )
        assert mention_compat_scores.shape == (batch_size, spans_nb, spans_nb)

        # -- final mention scores computation --
        #    s_m(m1) + s_m(m2) + s_c(m1, m2)
        final_scores = (
            # (batch_size, spans_nb, spans_nb)
            torch.stack([mention_scores for _ in range(spans_nb)], dim=1)
            # (batch_size, spans_nb, spans_nb)
            + torch.stack([mention_scores for _ in range(spans_nb)], dim=1).transpose(
                1, 2
            )
            # (batch_size, spans_nb, spans_nb)
            + mention_compat_scores
        )
        assert final_scores.shape == (batch_size, spans_nb, spans_nb)
        # (batch_size, spans_nb, spans_nb)
        # final_scores = torch.softmax(final_scores, dim=2)

        # TODO: final_scores and labels should be (spans_nb, spans_nb + 1)
        #       to account for the dummy antecedent. It seems that the
        #       score of the combination with the dummy antecedent should
        #       be 0.

        # -- loss computation --
        loss = None
        if labels is not None:
            loss = self.loss(final_scores, labels)

        return BertCoreferenceResolutionOutput(
            logits=final_scores,
            loss=loss,
            hidden_states=bert_output.hidden_states,
            attentions=bert_output.attentions,
        )

    def loss(self, pred_scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        :param pred_scores: ``(batch_size, spans_nb, spans_ nb)``
        :param labels: ``(batch_size, spans_nb, spans_ nb)``
        :return: ``(batch_size, spans_nb)``
        """
        # NOTE: reproduced from
        #       https://github.com/kentonl/e2e-coref/blob/master/coref_model.py
        #       /!\ more understanding is needed.
        batch_size = pred_scores.shape[0]
        spans_nb = pred_scores.shape[1]

        gold_scores = pred_scores + torch.log(labels)

        marginalized_gold_scores = torch.logsumexp(gold_scores, dim=2)
        assert marginalized_gold_scores.shape == (batch_size, spans_nb)

        log_norm = torch.logsumexp(pred_scores, dim=2)
        assert log_norm.shape == (batch_size, spans_nb)

        # (batch_size, spans_nb)
        return log_norm - marginalized_gold_scores
