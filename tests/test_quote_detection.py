import string
from typing import List, Tuple
from hypothesis import given, assume
import hypothesis.strategies as st
from more_itertools import flatten
from renard.pipeline.quote_detection import Quote, QuoteDetector


@given(
    prev_text=st.lists(st.text(alphabet=string.ascii_letters)),
    after_text=st.lists(st.text(alphabet=string.ascii_letters)),
    quote_content=st.lists(st.text(alphabet=string.ascii_letters)),
    quote_pair=st.sampled_from(QuoteDetector.DEFAULT_QUOTE_PAIRS),
)
def test_quote_is_extracted(
    prev_text: List[str],
    after_text: List[str],
    quote_content: List[str],
    quote_pair: Tuple[str, str],
):
    quote = [quote_pair[0]] + quote_content + [quote_pair[1]]
    text = prev_text + quote + after_text

    quote_detector = QuoteDetector()
    should_detect_quote = Quote(len(prev_text), len(text) - len(after_text), quote)

    detected = quote_detector(tokens=text)["quotes"]

    assert len(detected) == 1
    assert detected[0] == should_detect_quote


@given(text=st.lists(st.text()))
def test_quote_is_not_extracted(text: List[str]):
    all_quotes = list(flatten(QuoteDetector.DEFAULT_QUOTE_PAIRS))
    assume(all([not c in text for c in all_quotes]))
    quote_detector = QuoteDetector()
    assert len(quote_detector(tokens=text)["quotes"]) == 0
