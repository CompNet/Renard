from typing import Dict, Any, List, Optional, Set, Union, Literal
import itertools
import torch
import nltk
from transformers.tokenization_utils_base import BatchEncoding
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
