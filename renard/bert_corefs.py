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

        self.mention_compatibility_scorer = torch.nn.Linear(
            2 * 2 * config.hidden_size, 1
        )

    def mention_score(self, span_bounds: torch.Tensor) -> torch.Tensor:
        """Compute a score representing how likely it is that a span is a mention

        :param span_bounds: a tensor of shape ``(batch_size, 2, hidden_size)``,
            representing the first and last token of a span.

        :return: a tensor of shape ``(batch_size)``.
        """
        # (batch_size)
        return self.mention_scorer(torch.flatten(span_bounds, 1)).squeeze(-1)

    def mention_compatibility_score(
        self, first_span_bounds: torch.Tensor, second_span_bounds: torch.Tensor
    ) -> torch.Tensor:
        """
        :param first_span_bounds: ``(batch_size, 2, hidden_size)``
        :param second_span_bounds: ``(batch_size, 2, hidden_size)``

        :return: a tensor of shape ``(batch_size)``
        """
        return self.mention_compatibility_scorer(
            torch.flatten(torch.cat((first_span_bounds, second_span_bounds), 2), 1)
        ).squeeze(-1)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
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
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

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

        # let's suppose...
        # (batch_size, max_seq_size, hidden_size)
        encoded_input = bert_output[0]

        spans_idx = spans(range(bert_output[0].shape[1]), self.config.max_span_size)
        # (spans_nb, 2)
        spans_selector = torch.tensor(
            [[span[0], span[-1]] for span in spans_idx], dtype=torch.long
        )

        # TODO select spans in encoded_input to compute mention scores
