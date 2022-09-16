from typing import Callable, Dict, Any, List, Optional, Set
import torch
from more_itertools.recipes import flatten
from renard.pipeline.core import PipelineStep


class NLTKWordTokenizer(PipelineStep):
    """Construct a nltk word tokenizer"""

    def __init__(self, language="eng"):
        """:param language: language, passed to :func:`nltk.word_tokenize`"""
        self.language = language
        super().__init__()

    def __call__(
        self, text: str, chapters: Optional[List[str]] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        :param text:
        """
        import nltk

        nltk.download("punkt", quiet=True)

        if chapters is None:
            tokens = nltk.word_tokenize(text)
            return {"tokens": tokens, "chapter_tokens": [tokens]}

        chapter_tokens = [nltk.word_tokenize(c) for c in chapters]
        tokens = list(flatten(chapter_tokens))
        return {"tokens": tokens, "chapter_tokens": chapter_tokens}

    def needs(self) -> Set[str]:
        return {"text"}

    def production(self) -> Set[str]:
        return {"tokens", "chapter_tokens"}


class BertTokenizer(PipelineStep):
    """Tokenizer for bert based models"""

    def __init__(self, huggingface_model_id: str = "bert-base-cased") -> None:
        """
        :param huggingface_model_id:
        """
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(huggingface_model_id)
        super().__init__()

    def __call__(
        self, text: str, chapters: Optional[List[str]] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        :param text:
        """
        import nltk

        nltk.download("punkt", quiet=True)

        if chapters is None:
            chapters = [text]

        chapter_tokens = []
        wp_tokens = []
        batchs = {}
        for chapter in chapters:
            chapter_tokens.append([])
            batch = self.tokenizer(
                nltk.sent_tokenize(chapter),
                return_tensors="pt",
                padding=True,
                truncation=True,
            )
            if len(batchs) == 0:
                batchs = batch
            else:
                batchs = {k: torch.cat([v, batch[k]], dim=0) for k, v in batch.items()}
            nested_wp_tokens: List[List[str]] = [
                batch.tokens(i) for i in range(len(batch["input_ids"]))
            ]
            chapter_wp_tokens = [wp_t for s in nested_wp_tokens for wp_t in s]
            wp_tokens += chapter_wp_tokens
            chapter_tokens[-1] += BertTokenizer.wp_tokens_to_tokens(chapter_wp_tokens)

        return {
            "tokens": list(flatten(chapter_tokens)),
            "chapter_tokens": chapter_tokens,
            "bert_batch_encoding": batchs,
            "wp_tokens": wp_tokens,
        }

    def needs(self) -> Set[str]:
        return {"text"}

    def production(self) -> Set[str]:
        return {"tokens", "bert_batch_encoding", "wp_tokens", "chapter_tokens"}

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
