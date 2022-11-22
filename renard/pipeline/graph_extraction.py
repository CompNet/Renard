from typing import Dict, Any, List, Set, Optional, Tuple, Literal
import operator
from itertools import accumulate

import networkx as nx
import numpy as np
from more_itertools import windowed

from renard.pipeline.ner import NEREntity
from renard.pipeline.core import PipelineStep
from renard.pipeline.characters_extraction import Character


def sent_index_for_token_index(token_index: int, sentences: List[List[str]]) -> int:
    """Compute the index of the sentence of the token at ``token_index``"""
    sents_len = accumulate([len(s) for s in sentences], operator.add)
    return next((i for i, l in enumerate(sents_len) if l > token_index))


def sent_indices_for_chapter(
    chapters: List[List[str]], chapter_idx: int, sentences: List[List[str]]
) -> Tuple[int, int]:
    """Return the indices of the first and the last sentence of a
    chapter

    :param chapters: all chapters
    :param chapter_idx: index of the chapter for which sentence
        indices are returned
    :param sentences: all sentences
    :return: ``(first sentence index, last sentence index)``
    """
    chapter_start_idx = sum([len(c) for i, c in enumerate(chapters) if i < chapter_idx])
    chapter_end_idx = chapter_start_idx + len(chapters[chapter_idx])
    sents_start_indices = accumulate(
        [len(s) for s in sentences], operator.add, initial=0
    )
    return (
        next((i for i in sents_start_indices if i >= chapter_start_idx)),
        next((i for i in sents_start_indices if i >= chapter_end_idx)) - 1,
    )


def mentions_for_chapter(
    chapters: List[List[str]],
    chapter_idx: int,
    mentions: List[Tuple[Character, NEREntity]],
) -> List[Tuple[Character, NEREntity]]:
    """Return the mentions in the specified chapter

    :param chapters:
    :param chapter_idx: index of the specified chapter
    :param mentions:
    """
    chapter_start_idx = sum([len(c) for i, c in enumerate(chapters) if i < chapter_idx])
    chapter_end_idx = chapter_start_idx + len(chapters[chapter_idx])
    # TODO: optim
    return [
        m
        for m in mentions
        if m[1].start_idx >= chapter_start_idx and m[1].end_idx <= chapter_end_idx
    ]


class CoOccurencesGraphExtractor(PipelineStep):
    """A simple character graph extractor using co-occurences"""

    def __init__(
        self,
        co_occurences_dist: int,
        dynamic: Optional[Literal["nx", "gephi"]] = None,
        dynamic_window: Optional[int] = None,
        dynamic_overlap: int = 0,
    ) -> None:
        """
        :param co_occurences_dist: max accepted distance between two
            character appearances to form a co-occurence interaction,
            in number of tokens.

        :param dynamic: either ``None``, or one of ``{'nx', 'gephi'}``

                - if ``None`` (the defaul), a ``nx.graph`` is
                  extracted

                - if ``'nx'``, several ``nx.graph`` are extracted.  In
                  that case, ``dynamic_window`` and
                  ``dynamic_overlap``*can* be specified.  If
                  ``dynamic_window`` is not specified, this step is
                  expecting the text to be cut into chapters', and a
                  graph will be extracted for each 'chapter'.  In that
                  case, ``chapters`` must be passed to the pipeline as
                  a ``List[str]`` at runtime.

                - if ``'gephi'``, a single ``nx.graph`` is extracted.
                  This graph has the nice property that exporting it
                  to `gexf` format using ``G.write_gexf()`` will
                  produce a correct dynamic graph that can be read by
                  Gephi.  Because of a limitation in networkx, the
                  dynamic weight attribute is stored as ``dweight``
                  instead of ``weight``.

        :param dynamic_window: dynamic window, in number of
            interactions.  a dynamic window of `n` means that each
            returned graph will be formed by `n` interactions.

        :param dynamic_overlap: overlap, in number of interactions.
        """

        self.co_occurences_dist = co_occurences_dist

        if not dynamic is None:
            assert dynamic in {"nx", "gephi"}
            if dynamic == "nx":
                if not dynamic_window is None:
                    assert dynamic_window > 0
                    assert dynamic_window > dynamic_overlap
        self.dynamic = dynamic
        self.dynamic_window = dynamic_window
        self.dynamic_overlap = dynamic_overlap
        self.dynamic_needs_chapter = dynamic == "nx" and dynamic_window is None
        super().__init__()

    def __call__(
        self,
        text: str,
        tokens: List[str],
        bio_tags: List[str],
        characters: Set[Character],
        chapter_tokens: Optional[List[List[str]]] = None,
        sentences: Optional[List[List[str]]] = None,
        sentences_polarities: Optional[List[float]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Extract a characters graph

        :param tokens:
        :param bio_tags:
        :param characters:

        :return: a ``dict`` with key ``'characters_graph'`` and a
            ``networkx.Graph`` or a list of ``networkx.Graph`` as
            value.
        """
        assert len(tokens) == len(bio_tags)

        mentions = []
        for character in characters:
            for mention in character.mentions:
                mentions.append((character, mention))
        mentions = sorted(mentions, key=lambda cm: cm[1].start_idx)

        if self.dynamic == "gephi":
            if not sentences is None and not sentences_polarities is None:
                print("[warning] 'gephi' does not support sentence polarities")
            return {"characters_graph": self._extract_gephi_dynamic_graph(mentions)}

        elif self.dynamic == "nx":
            return {
                "characters_graph": self._extract_dynamic_graph(
                    mentions,
                    self.dynamic_window,
                    self.dynamic_overlap,
                    chapter_tokens,
                    sentences,
                    sentences_polarities,
                )
            }

        # static extraction
        return {
            "characters_graph": self._extract_graph(
                mentions, sentences, sentences_polarities
            )
        }

    def _extract_graph(
        self,
        mentions: List[Tuple[Character, NEREntity]],
        sentences: Optional[List[List[str]]],
        sentences_polarities: Optional[List[float]],
    ):
        """
        :param mentions: A list of character mentions, ordered by
            appearance
        :param sentences: if specified, ``sentences_polarities`` must
            be specified as well.
        :param sentences_polarities: if specified, ``sentences`` must
            be specified as well.  In that case, edges are annotated
            with the ``'polarity`` attribute, indicating the polarity
            of the relationship between two characters.  Polarity
            between two interactions is computed as the strongest
            sentence polarity between those two mentions.
        """
        compute_polarity = not sentences is None and not sentences_polarities is None

        # co-occurence matrix, where C[i][j] is 1 when appearance
        # i co-occur with j if i < j, or 0 when it doesn't
        C = np.zeros((len(mentions), len(mentions)))
        for i, (char1, mention_1) in enumerate(mentions):
            # check ahead for co-occurences
            for j, (char2, mention_2) in enumerate(mentions[i + 1 :]):
                if mention_2.start_idx - mention_1.start_idx > self.co_occurences_dist:
                    # dist between current token and future token is
                    # too great : we finished co-occurences search for
                    # the current token
                    break
                # ignore co-occurences with self
                if char1 == char2:
                    continue
                # record co_occurence
                C[i][i + 1 + j] = 1

        # construct graph from co-occurence matrix
        G = nx.Graph()

        for i, (char1, mention1) in enumerate(mentions):

            for j, (char2, mention2) in enumerate(mentions):

                # no co-occurences for these two mentions: we are out
                if C[i][j] == 0:
                    continue

                if not G.has_edge(char1, char2):
                    G.add_edge(char1, char2, weight=0)
                G.edges[char1, char2]["weight"] += 1

                if compute_polarity:
                    assert not sentences is None
                    assert not sentences_polarities is None
                    # TODO: optim
                    first_sent_idx = sent_index_for_token_index(
                        mention1.start_idx, sentences
                    )
                    last_sent_idx = sent_index_for_token_index(
                        mention2.start_idx, sentences
                    )
                    sents_polarities_between_mentions = sentences_polarities[
                        first_sent_idx : last_sent_idx + 1
                    ]
                    polarity = max(sents_polarities_between_mentions, key=abs)
                    G.edges[char1, char2]["polarity"] = (
                        G.edges[char1, char2].get("polarity", 0) + polarity
                    )

        return G

    def _extract_dynamic_graph(
        self,
        mentions: List[Tuple[Character, NEREntity]],
        window: Optional[int],
        overlap: int,
        chapter_tokens: Optional[List[List[str]]],
        sentences: Optional[List[List[str]]],
        sentences_polarities: Optional[List[float]],
    ) -> List[nx.Graph]:
        """
        .. note::

            only one of ``window`` or ``chapter_tokens`` should be specified

        :param mentions: A list of character mentions, ordered by appearance
        :param window: dynamic window, in tokens.
        :param overlap: window overlap
        :param chapter_tokens: list of tokens for each chapter.  If
            given, one graph will be extracted per chapter.
        """
        assert window is None or chapter_tokens is None
        compute_polarity = not sentences is None and not sentences_polarities is None

        if not window is None:
            return [
                self._extract_graph(
                    [elt for elt in ct if not elt is None],
                    sentences,
                    sentences_polarities,
                )
                for ct in windowed(mentions, window, step=window - overlap)
            ]

        assert not chapter_tokens is None

        graphs = []

        for chapter_i, chapter in enumerate(chapter_tokens):

            # TODO: optim
            mentions = mentions_for_chapter(chapter_tokens, chapter_i, mentions)

            chapter_sentences = None
            chapter_sentences_polarities = None
            if compute_polarity:
                assert not sentences is None
                assert not sentences_polarities is None
                sent_start_idx, sent_end_idx = sent_indices_for_chapter(
                    chapter_tokens, chapter_i, sentences
                )
                chapter_sentences = sentences[sent_start_idx : sent_end_idx + 1]
                chapter_sentences_polarities = sentences_polarities[
                    sent_start_idx : sent_end_idx + 1
                ]

            graphs.append(
                self._extract_graph(
                    mentions,
                    chapter_sentences,
                    chapter_sentences_polarities,
                )
            )

        return graphs

    def _extract_gephi_dynamic_graph(
        self, mentions: List[Tuple[Character, NEREntity]]
    ) -> nx.Graph:
        """
        :param mentions: A list of character mentions, ordered by appearance
        """
        # keep only longest name in graph node : possible only if it is unique
        # TODO: might want to try and get shorter names if longest names aren't
        #       unique
        characters = set([e[0] for e in mentions])

        G = nx.Graph()

        character_to_last_appearance: Dict[Character, Optional[int]] = {
            character: None for character in characters
        }

        for i, (character, mention) in enumerate(mentions):
            if not character in characters:
                continue
            character_to_last_appearance[character] = mention.start_idx
            close_characters = [
                c
                for c, last_appearance in character_to_last_appearance.items()
                if not last_appearance is None
                and mention.start_idx - last_appearance <= self.co_occurences_dist
                and not c == character
            ]
            for close_character in close_characters:
                if not G.has_edge(character, close_character):
                    G.add_edge(character, close_character)
                    G.edges[character, close_character]["start"] = i
                    G.edges[character, close_character]["dweight"] = []
                # add a new entry to the weight series according to networkx
                # source code, each entry must be of the form
                # [value, start, end]
                weights = G.edges[character, close_character]["dweight"]
                if len(weights) != 0:
                    # end of last weight attribute
                    weights[-1][-1] = i
                # value, start and end of current weight attribute
                last_weight_value = weights[-1][0] if len(weights) > 0 else 0
                G.edges[character, close_character]["dweight"].append(
                    [float(last_weight_value) + 1, i, len(mentions)]
                )

        return G

    def needs(self) -> Set[str]:
        needs = {"tokens", "bio_tags", "characters"}
        if self.dynamic_needs_chapter:
            needs.add("chapter_tokens")
        return needs

    def production(self) -> Set[str]:
        return {"characters_graph"}

    def optional_needs(self) -> Set[str]:
        return {"sentences_polarities", "sentences"}
