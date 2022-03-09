from typing import Optional
import torch
from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertModel
from transformers.models.bert.configuration_bert import BertConfig
from renard.utils import spans


class BertForCoreferenceResolution(BertPreTrainedModel):
    """"""

    def __init__(self, config: BertConfig):
        super().__init__(config)

        self.bert = BertModel(config, add_pooling_layer=False)

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
        :param span_bounds: ``(batch_size, 4, hidden_size)``

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
        input_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        """
        TODO: add return type

        :param input_ids:
        :param attention_mask:
        """
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        batch_size = input_ids.shape[0]
        seq_size = input_ids.shape[1]
        hidden_size = self.config.hidden_size

        bert_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            input_embeds=input_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # (batch_size, max_seq_size, hidden_size)
        encoded_input = bert_output.last_hidden_state

        # -- span bounds computation --
        # we select starting and ending bounds of spans of length up
        # to self.config.max_span_size
        spans_idx = spans(range(seq_size), self.config.max_span_size)
        spans_nb = len(spans_idx)
        # (spans_nb * 2)
        spans_selector = torch.flatten(
            torch.tensor([[span[0], span[-1]] for span in spans_idx], dtype=torch.long)
        )
        # (batch_size, spans_nb, 2, hidden_size)
        span_bounds = torch.index_select(encoded_input, 1, spans_selector).reshape(
            batch_size, spans_nb, 2, hidden_size
        )

        # -- mention scores computation --
        # (batch_size * spans_nb)
        mention_scores = self.mention_score(
            torch.flatten(span_bounds, start_dim=0, end_dim=1)
        )
        # (batch_size, spans_nb)
        mention_scores = mention_scores.reshape((batch_size, spans_nb))

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
        # (batch_size, spans_nb ** 2, 4, hidden_size)
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
        # (batch_size * spans_nb ** 2)
        mention_compat_scores = self.mention_compatibility_score(
            torch.flatten(span_bounds_combination, start_dim=0, end_dim=1)
        )
        # (batch_size, spans_nb, spans_nb)
        mention_compat_scores = mention_compat_scores.reshape(
            (batch_size, spans_nb, spans_nb)
        )

        # -- final mention scores computation --
        # (batch_size, spans_nb, spans_nb)
        final_scores = (
            torch.stack([mention_scores for _ in range(spans_nb)], dim=1)
            + torch.stack([mention_scores for _ in range(spans_nb)], dim=1).T
            + mention_compat_scores
        )
        # (batch_size, spans_nb, spans_nb)
        final_scores = torch.softmax(final_scores, dim=2)
