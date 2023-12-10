from typing import List, Literal, Optional, Union
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding


class DataCollatorForTokenClassificationWithBatchEncoding:
    """Same as ``transformers.DataCollatorForTokenClassification``,
    except it correctly returns a ``BatchEncoding`` object with
    correct ``encodings`` attribute.

    Don't know why this is not the default ?
    """

    def __init__(
        self,
        tokenizer: PreTrainedTokenizerFast,
        pad_to_multiple_of: Optional[int] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.pad_token_id = {"label": -100, "labels": -100}

    def __call__(self, features: List[dict]) -> Union[dict, BatchEncoding]:
        keys = features[0].keys()
        sequence_len = max([len(f["input_ids"]) for f in features])

        # We do the padding and collating manually instead of calling
        # self.tokenizer.pad, because pad does not work on arbitrary
        # features.
        batch = BatchEncoding({})
        for key in keys:
            if self.tokenizer.padding_side == "right":
                batch[key] = [
                    f[key]
                    + [self.pad_token_id.get(key, 0)] * (sequence_len - len(f[key]))
                    for f in features
                ]
            else:
                batch[key] = [
                    [
                        self.pad_token_id.get(key, 0) * (sequence_len - len(f[key]))
                        + f[key]
                        for f in features
                    ]
                ]

        batch._encodings = [f.encodings[0] for f in features]

        for k, v in batch.items():
            batch[k] = torch.tensor(v)

        return batch


class NERDataset(Dataset):
    """
    :ivar _context_mask: for each element, a mask indicating which
        tokens are part of the context (1 for context, 0 for text on
        which to perform inference).  The mask allows to discard
        predictions made for context at inference time, even though
        the context can still be passed as input to the model.
    """

    def __init__(
        self,
        elements: List[List[str]],
        tokenizer: PreTrainedTokenizerFast,
        context_mask: Optional[List[List[int]]] = None,
    ) -> None:
        self.elements = elements

        if context_mask:
            assert all(
                [len(cm) == len(elt) for elt, cm in zip(self.elements, context_mask)]
            )
        self._context_mask = context_mask or [[0] * len(elt) for elt in self.elements]

        self.tokenizer = tokenizer

    def __getitem__(self, index) -> BatchEncoding:
        element = self.elements[index]

        batch = self.tokenizer(
            element,
            truncation=True,
            max_length=512,  # TODO
            is_split_into_words=True,
        )

        batch["context_mask"] = [0] * len(batch["input_ids"])
        elt_context_mask = self._context_mask[index]
        for i in range(len(element)):
            w2t = batch.word_to_tokens(0, i)
            mask_value = elt_context_mask[i]
            tokens_mask = [mask_value] * (w2t.end - w2t.start)
            batch["context_mask"][w2t.start : w2t.end] = tokens_mask

        return batch

    def __len__(self) -> int:
        return len(self.elements)
