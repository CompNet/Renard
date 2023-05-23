from typing import Dict, Any, List, Optional, Set, Union, Literal
import itertools
import torch
import nltk
from more_itertools.recipes import flatten
from renard.pipeline.core import PipelineStep
from renard.pipeline.progress import ProgressReporter
from renard.nltk_utils import NLTK_ISO_STRING_TO_LANG


class NLTKTokenizer(PipelineStep):
    """Construct a nltk word tokenizer"""

    def __init__(self):
        nltk.download("punkt", quiet=True)
        super().__init__()

    def __call__(
        self, text: str, chapters: Optional[List[str]] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        :param text:
        """
        if not chapters is None:
            out_dicts = [self.__call__(chapter, None) for chapter in chapters]
            return {
                "tokens": list(itertools.chain([d["tokens"] for d in out_dicts])),
                "sentences": list(itertools.chain([d["sentences"] for d in out_dicts])),
                "chapter_tokens": [d["tokens"] for d in out_dicts],
            }

        sentences = nltk.sent_tokenize(
            text, language=NLTK_ISO_STRING_TO_LANG[self.lang]
        )

        tokens = []
        tokenized_sentences = []
        for sent in sentences:
            sent_tokens = nltk.word_tokenize(sent)
            tokenized_sentences.append(sent_tokens)
            tokens += sent_tokens

        return {"tokens": tokens, "sentences": tokenized_sentences}

    def supported_langs(self) -> Union[Set[str], Literal["any"]]:
        return set(NLTK_ISO_STRING_TO_LANG.keys())

    def needs(self) -> Set[str]:
        return {"text"}

    def production(self) -> Set[str]:
        return {"tokens", "chapter_tokens", "sentences"}


class BertTokenizer(PipelineStep):
    """Tokenizer for bert based models

    .. note::

        While this tokenizer produces tokens using
        a word piece tokenizer, ``sentences`` are
        obtained using NLTK's sentence tokenizer.
    """

    def __init__(self, huggingface_model_id: Optional[str] = None) -> None:
        """
        :param huggingface_model_id: A custom huggingface model id.
            This allows to bypass the ``lang`` pipeline parameter,
            which normally choose a huggingface model automatically.
        """
        self.huggingface_model_id = huggingface_model_id
        nltk.download("punkt", quiet=True)
        super().__init__()

    def _pipeline_init_(self, lang: str, progress_reporter: ProgressReporter):
        from transformers import AutoTokenizer

        super()._pipeline_init_(lang, progress_reporter)

        if not self.huggingface_model_id is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.huggingface_model_id)
            self.lang = "unknown"
        else:
            if lang == "eng":
                self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
            elif lang == "fra":
                self.tokenizer = AutoTokenizer.from_pretrained("camembert-base")
            else:
                raise ValueError(f"BertTokenizer does not support language {lang}")

    def __call__(
        self, text: str, chapters: Optional[List[str]] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        :param text:
        """
        if not chapters is None:
            out_dicts = [self.__call__(chapter, None) for chapter in chapters]
            out_batchs = [d["bert_batch_encoding"] for d in out_dicts]
            batchs = out_batchs[0]
            for b in batchs[1:]:
                batchs = {k: torch.cat([v, b[k]], dim=0) for k, v in batchs.items()}
            return {
                "tokens": list(itertools.chain([d["tokens"] for d in out_dicts])),
                "sentences": list(itertools.chain([d["sentences"] for d in out_dicts])),
                "chapter_tokens": [d["tokens"] for d in out_dicts],
                "wp_tokens": list(itertools.chain([d["tokens"] for d in out_dicts])),
                "bert_batch_encoding": batchs,
            }

        # NOTE: it's possible that some input tokens are discarded
        # here because of truncation.
        batchs = self.tokenizer(
            nltk.sent_tokenize(text, language=NLTK_ISO_STRING_TO_LANG[self.lang]),
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        nested_wp_tokens: List[List[str]] = [
            batchs.tokens(i) for i in range(len(batchs["input_ids"]))
        ]
        wp_tokens = [wp_t for s in nested_wp_tokens for wp_t in s]

        sentences = [
            BertTokenizer.wp_tokens_to_tokens(wp_tokens)
            for wp_tokens in nested_wp_tokens
        ]
        tokens = list(flatten(sentences))

        return {
            "tokens": tokens,
            "sentences": sentences,
            "bert_batch_encoding": batchs,
            "wp_tokens": wp_tokens,
        }

    def supported_langs(self) -> Union[Set[str], Literal["any"]]:
        return {"eng", "fra"}

    def needs(self) -> Set[str]:
        return {"text"}

    def production(self) -> Set[str]:
        return {
            "tokens",
            "bert_batch_encoding",
            "sentences",
            "wp_tokens",
            "chapter_tokens",
        }

    @staticmethod
    def wp_tokens_to_tokens(wp_tokens: List[str]) -> List[str]:
        """Convert word piece tokens to 'regular' tokens

        :wp_tokens: word piece tokens
        """
        tokens = []
        for wp_token in wp_tokens:
            if wp_token in {"[CLS]", "[SEP]", "[PAD]"}:
                continue
            if not wp_token.startswith("##"):
                tokens.append(wp_token)
            else:
                tokens[-1] += wp_token[2:]
        return tokens
