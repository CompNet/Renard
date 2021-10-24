from typing import Callable, Dict, Any, List, Set
from pipeline import PipelineStep


class NLTKWordTokenizer(PipelineStep):
    """Construct a nltk word tokenizer

    :ivar language: language, passed to :func:`nltk.word_tokenize`
    """

    def __init__(self, language="eng"):
        self.language = language

    def __call__(self, text: str, **kwargs) -> Dict[str, Any]:
        import nltk

        nltk.download("punkt", quiet=True)

        return {"tokens": nltk.word_tokenize(text)}

    def needs(self) -> Set[str]:
        return {"text"}

    def produces(self) -> Set[str]:
        return {"tokens"}


class BertTokenizer(PipelineStep):
    """Tokenizer for bert based models"""

    def __init__(self, huggingface_model_id: str = "bert-base-cased") -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(huggingface_model_id)

    def __call__(self, text: str, **kwargs) -> Dict[str, Any]:
        import nltk

        nltk.download("punkt", quiet=True)

        input_dict = self.tokenizer(
            nltk.sent_tokenize(text), return_tensors="pt", padding=True, truncation=True
        )
        nested_wp_tokens: List[List[str]] = [
            input_dict.tokens(i) for i in range(len(input_dict["input_ids"]))
        ]
        wp_tokens = [wp_t for s in nested_wp_tokens for wp_t in s]
        tokens = BertTokenizer.wp_tokens_to_tokens(wp_tokens)
        return {
            "tokens": tokens,
            "bert_batch_encoding": input_dict,
            "wp_tokens": wp_tokens,
        }

    def needs(self) -> Set[str]:
        return {"text"}

    def produces(self) -> Set[str]:
        return {"tokens", "bert_batch_encoding", "wp_tokens"}

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
