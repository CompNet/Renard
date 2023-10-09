from typing import List, Optional, Set, Dict, Any, Tuple, Union, Literal
from dataclasses import dataclass
from renard.pipeline.core import PipelineStep


@dataclass
class Quote:
    start: int
    end: int
    #: tokens, including both quote characters
    tokens: List[str]

    def tokens_without_quotes(self) -> List[str]:
        return self.tokens[1:-1]


class QuoteDetector(PipelineStep):
    """Extract quotes using simple rules."""

    DEFAULT_QUOTE_PAIRS = [('"', '"'), ("``", "''"), ("«", "»"), ("“", "”")]

    def __init__(self, quote_pairs: Optional[List[Tuple[str, str]]] = None):
        """
        :param quote_pairs: if ``None``, default to
            ``QuoteDetector.DEFAULT_QUOTE_PAIRS``
        """
        self.quote_pairs = quote_pairs or QuoteDetector.DEFAULT_QUOTE_PAIRS
        super().__init__()

    def _get_quote_pair(self, quote: str) -> Optional[Tuple[str, str]]:
        for qp in self.quote_pairs:
            if quote == qp[0] or quote == qp[1]:
                return qp
        return None

    def __call__(self, tokens: List[str], **kwargs) -> Dict[str, Any]:
        quotes = []
        cur_quote = None

        for token_i, token in enumerate(tokens):

            if not cur_quote is None:
                cur_quote.tokens.append(token)

            qp = self._get_quote_pair(token)
            if qp is None:
                continue

            is_opening_quote = token == qp[0]

            if is_opening_quote and cur_quote is None:
                cur_quote = Quote(token_i, -1, [token])
            else:
                if not cur_quote is None:
                    cur_quote.end = token_i + 1
                    quotes.append(cur_quote)
                    cur_quote = None

        return {"quotes": quotes}

    def needs(self) -> Set[str]:
        """tokens"""
        return {"tokens"}

    def production(self) -> Set[str]:
        """quotes"""
        return {"quotes"}

    def supported_langs(self) -> Union[Set[str], Literal["any"]]:
        """any"""
        return "any"
