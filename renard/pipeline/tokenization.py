from typing import Dict, Any, Set, Union, Literal, List, Tuple
from more_itertools import windowed
import nltk
from nltk.data import load
from nltk.tokenize.destructive import NLTKWordTokenizer
from renard.pipeline.core import PipelineStep
from renard.nltk_utils import NLTK_ISO_STRING_TO_LANG


def make_char2token(text: str, token2chars: List[Tuple[int, int]]) -> List[int]:
    if len(token2chars) == 0:
        return []

    c2t = [None] * len(text)
    for token_i, chars in enumerate(token2chars):
        for char_i in range(*chars):
            c2t[char_i] = token_i  # type: ignore

    for char_i in range(0, token2chars[0][0]):
        c2t[char_i] = 0  # type: ignore
    for chars1, chars2 in windowed(token2chars, 2):
        if chars1 is None or chars2 is None:
            continue
        end1 = chars1[1]
        start2 = chars2[0]
        for char_i in range(end1, start2):
            c2t[char_i] = c2t[end1 - 1]
    for char_i in range(token2chars[-1][1], len(c2t)):
        c2t[char_i] = token2chars[-1][1]  # type: ignore

    assert all([not i is None for i in c2t])
    return c2t  # type: ignore


class NLTKTokenizer(PipelineStep):
    """A NLTK-based tokenizer"""

    def __init__(self):
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        self.word_tokenizer = None
        self.sent_tokenizer = None
        super().__init__()

    def _pipeline_init_(self, lang: str, **kwargs):
        assert lang in NLTK_ISO_STRING_TO_LANG
        nltk_lang = NLTK_ISO_STRING_TO_LANG[lang]
        self.word_tokenizer = NLTKWordTokenizer()
        self.sent_tokenizer = load(f"tokenizers/punkt/{nltk_lang}.pickle")
        super()._pipeline_init_(lang, **kwargs)

    def __call__(self, text: str, **kwargs) -> Dict[str, Any]:
        assert not self.word_tokenizer is None
        assert not self.sent_tokenizer is None

        sent_indices = self.sent_tokenizer.span_tokenize(text)

        tokens = []
        token2chars = []
        tokenized_sentences = []
        for sent_start, sent_end in sent_indices:
            sent = text[sent_start:sent_end]
            sent_tokens_indices = list(self.word_tokenizer.span_tokenize(sent))
            token2chars += [
                (start + sent_start, end + sent_start)
                for start, end in sent_tokens_indices
            ]
            sent_tokens = [sent[start:end] for start, end in sent_tokens_indices]
            tokenized_sentences.append(sent_tokens)
            tokens += sent_tokens

        return {
            "tokens": tokens,
            "char2token": make_char2token(text, token2chars),
            "sentences": tokenized_sentences,
        }

    def supported_langs(self) -> Union[Set[str], Literal["any"]]:
        return set(NLTK_ISO_STRING_TO_LANG.keys())

    def needs(self) -> Set[str]:
        return {"text"}

    def production(self) -> Set[str]:
        return {"tokens", "char2token", "sentences"}
