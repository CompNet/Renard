from typing import List, Optional, Union
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
        self.label_pad_token_id = -100

    def __call__(self, features) -> Union[dict, BatchEncoding]:
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = (
            [feature[label_name] for feature in features]
            if label_name in features[0].keys()
            else None
        )
        batch = self.tokenizer.pad(
            features,
            padding="longest",
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt" if labels is None else None,
        )
        # keep encodings info dammit
        batch._encodings = [f.encodings[0] for f in features]

        if labels is None:
            return batch

        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side
        if padding_side == "right":
            batch[label_name] = [
                list(label) + [self.label_pad_token_id] * (sequence_length - len(label))
                for label in labels
            ]
        else:
            batch[label_name] = [
                [self.label_pad_token_id] * (sequence_length - len(label)) + list(label)
                for label in labels
            ]

        return batch


class NERDataset(Dataset):
    def __init__(
        self, sentences: List[List[str]], tokenizer: PreTrainedTokenizerFast
    ) -> None:
        self.sentences = sentences
        self.tokenizer = tokenizer

    def __getitem__(self, index) -> BatchEncoding:
        batch = self.tokenizer(
            self.sentences[index],
            return_tensors="pt",
            padding=True,
            truncation=True,
            is_split_into_words=True,
        )
        for k in batch.keys():
            batch[k] = batch[k][0]
        return batch

    def __len__(self) -> int:
        return len(self.sentences)
