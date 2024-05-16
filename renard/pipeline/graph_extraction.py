from typing import Dict, Any, List, Set, Optional, Tuple, Literal, Union
import itertools as it
import operator

import networkx as nx
import numpy as np
from more_itertools import windowed, flatten

from renard.pipeline.ner import NEREntity
from renard.pipeline.core import PipelineStep
from renard.pipeline.character_unification import Character
from renard.pipeline.quote_detection import Quote


def sent_index_for_token_index(token_index: int, sentences: List[List[str]]) -> int:
    """Compute the index of the sentence of the token at ``token_index``"""
    sents_len = it.accumulate([len(s) for s in sentences], operator.add)
    return next((i for i, l in enumerate(sents_len) if l > token_index))


def sent_indices_for_block(
    dynamic_blocks: List[List[str]], block_i: int, sentences: List[List[str]]
) -> Tuple[int, int]:
    """Return the indices of the first and the last sentence of a
    block

    :param dynamic_blocks: all blocks
    :param block_idx: index of the block for which sentence
        indices are returned
    :param sentences: all sentences
    :return: ``(first sentence index, last sentence index)``
    """
    block_start = sum([len(c) for i, c in enumerate(dynamic_blocks) if i < block_i])
    block_end = block_start + len(dynamic_blocks[block_i])
    sents_start = None
    sents_end = None
    count = 0
    for sent_i, sent in enumerate(sentences):
        start, end = (count, count + len(sent))
        count = end
        if sents_start is None and start >= block_start:
            sents_start = sent_i
        if sents_end is None and end >= block_end:
            sents_end = sent_i
            break
    assert not sents_start is None and not sents_end is None
    return (sents_start, sents_end)


def mentions_for_blocks(
    dynamic_blocks: List[List[str]],
    mentions: List[Tuple[Any, NEREntity]],
) -> List[List[Tuple[Any, NEREntity]]]:
    """Return each block mentions

    :param blocks:
    :param mentions:

    :return: a list of mentions per blocks.  This list has len
             ``len(blocks)``.
    """
    blocks_mentions = [[] for _ in range(len(dynamic_blocks))]

    start_indices = list(
        it.accumulate([0] + [len(block) for block in dynamic_blocks[:-1]])
    )
    end_indices = start_indices[1:] + [start_indices[-1] + len(dynamic_blocks[-1])]

    for mention in mentions:
        for block_i, (start_i, end_i) in enumerate(zip(start_indices, end_indices)):
            if mention[1].start_idx >= start_i and mention[1].end_idx < end_i:
                blocks_mentions[block_i].append(mention)
                break

    return blocks_mentions


class CoOccurrencesGraphExtractor(PipelineStep):
    """A simple character graph extractor using co-occurences"""

    def __init__(
        self,
        co_occurrences_dist: Optional[
            Union[int, Tuple[int, Literal["tokens", "sentences"]]]
        ] = None,
        dynamic: bool = False,
        dynamic_window: Optional[int] = None,
        dynamic_overlap: int = 0,
        additional_ner_classes: Optional[List[str]] = None,
    ) -> None:
        """
        :param co_occurrences_dist: max accepted distance between two
            character appearances to form a co-occurence interaction.

                - if an ``int`` is given, the distance is in number of
                  tokens

                - if a ``tuple`` is given, the first element of the
                  tuple is a distance while the second is an unit.
                  Examples : ``(1, "sentences")``, ``(3, "tokens")``.

        :param dynamic:

                - if ``False`` (the default), a static ``nx.graph`` is
                  extracted

                - if ``True``, several ``nx.graph`` are extracted.  In
                  that case, ``dynamic_window`` and
                  ``dynamic_overlap``*can* be specified.  If
                  ``dynamic_window`` is not specified, this step is
                  expecting the text to be cut into 'dynamic blocks',
                  and a graph will be extracted for each block.  In
                  that case, ``dynamic_blocks`` must be passed to the
                  pipeline as a ``List[str]`` at runtime.

        :param dynamic_window: dynamic window, in number of
            interactions.  a dynamic window of `n` means that each
            returned graph will be formed by `n` interactions.

        :param dynamic_overlap: overlap, in number of interactions.

        :param additional_ner_classes: if specified, will include
            entities other than characters in the final graph.  No
            attempt will be made at unifying the entities (for example,
            "New York" will be distinct from "New York City").
        """
        if isinstance(co_occurrences_dist, int):
            co_occurrences_dist = (co_occurrences_dist, "tokens")
        self.co_occurrences_dist = co_occurrences_dist

        if dynamic:
            if not dynamic_window is None:
                assert dynamic_window > 0
                assert dynamic_window > dynamic_overlap
        self.dynamic = dynamic
        self.dynamic_window = dynamic_window
        self.dynamic_overlap = dynamic_overlap
        self.need_dynamic_blocks = dynamic == "nx" and dynamic_window is None

        self.additional_ner_classes = additional_ner_classes or []

        super().__init__()

    def __call__(
        self,
        characters: Set[Character],
        sentences: List[List[str]],
        dynamic_blocks_tokens: Optional[List[List[str]]] = None,
        sentences_polarities: Optional[List[float]] = None,
        entities: Optional[List[NEREntity]] = None,
        co_occurrences_blocks: Optional[List[Tuple[int, int]]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Extract a characters graph

        :param co_occurrences_blocks: a list of tuple, each of the
            form (BLOCK_START_INDEX, BLOCK_END_INDEX).  custom blocks
            where co-occurrences should be recorded.  For example,
            this can be used to perform chapter level co-occurrences.

        :return: a ``dict`` with key ``'character_network'`` and a
                 :class:`nx.Graph` or a list of :class:`nx.Graph` as
                 value.
        """
        mentions = []
        for character in characters:
            for mention in character.mentions:
                mentions.append((character, mention))

        if len(self.additional_ner_classes) > 0:
            assert not entities is None
            for entity in entities:
                if entity.tag in self.additional_ner_classes:
                    mentions.append((" ".join(entity.tokens), entity))

        mentions = sorted(mentions, key=lambda cm: cm[1].start_idx)

        if self.dynamic:
            return {
                "character_network": self._extract_dynamic_graph(
                    mentions,
                    self.dynamic_window,
                    self.dynamic_overlap,
                    dynamic_blocks_tokens,
                    sentences,
                    sentences_polarities,
                    co_occurrences_blocks,
                )
            }
        return {
            "character_network": self._extract_graph(
                mentions, sentences, sentences_polarities, co_occurrences_blocks
            )
        }

    def _create_co_occurrences_blocks(
        self, sentences: List[List[str]], mentions: List[Tuple[Any, NEREntity]]
    ) -> List[Tuple[int, int]]:
        """Create co-occurrences blocks using
        ``self.co_occurrences_dist``.  All entities within a block are
        considered as co-occurring.

        :param sentences:
        """
        assert not self.co_occurrences_dist is None

        dist_unit = self.co_occurrences_dist[1]

        if dist_unit == "tokens":
            tokens_dist = self.co_occurrences_dist[0]
            blocks = []
            for _, entity in mentions:
                block_start = entity.start_idx - tokens_dist
                block_end = entity.end_idx + tokens_dist
                blocks.append((block_start, block_end))
            return blocks

        elif dist_unit == "sentences":
            blocks_indices = set()
            sent_dist = self.co_occurrences_dist[0]
            for _, entity in mentions:
                start_sent_i = max(
                    0,
                    sent_index_for_token_index(entity.start_idx, sentences) - sent_dist,
                )
                start_token_i = sum(len(sent) for sent in sentences[:start_sent_i])
                end_sent_i = min(
                    len(sentences) - 1,
                    sent_index_for_token_index(entity.end_idx - 1, sentences)
                    + sent_dist,
                )
                end_token_i = sum(len(sent) for sent in sentences[: end_sent_i + 1])
                blocks_indices.add((start_token_i, end_token_i))
            return [
                (start, end)
                for start, end in sorted(blocks_indices, key=lambda indices: indices[0])
            ]

        else:
            raise ValueError(
                f"co_occurrences_dist unit should be one of: 'tokens', 'sentences'"
            )

    def _extract_graph(
        self,
        mentions: List[Tuple[Any, NEREntity]],
        sentences: List[List[str]],
        sentences_polarities: Optional[List[float]],
        co_occurrences_blocks: Optional[List[Tuple[int, int]]],
    ):
        """
        :param mentions: A list of entity mentions, ordered by
            appearance, each of the form (KEY MENTION).  KEY
            determines the unicity of the entity.
        :param sentences: if specified, ``sentences_polarities`` must
            be specified as well.
        :param sentences_polarities: if specified, ``sentences`` must
            be specified as well.  In that case, edges are annotated
            with the ``'polarity`` attribute, indicating the polarity
            of the relationship between two characters.  Polarity
            between two interactions is computed as the strongest
            sentence polarity between those two mentions.
        """
        compute_polarity = not sentences_polarities is None
        if co_occurrences_blocks is None:
            co_occurrences_blocks = self._create_co_occurrences_blocks(
                sentences, mentions
            )

        # co-occurence matrix, where C[i][j] is 1 when appearance
        # i co-occur with j if i < j, or 0 when it doesn't
        C = np.zeros((len(mentions), len(mentions)))
        for block_start, block_end in co_occurrences_blocks:
            # collect all mentions in this co-occurrences block
            block_mentions = []
            for i, (key, mention) in enumerate(mentions):
                if mention.start_idx >= block_start and mention.end_idx <= block_end:
                    block_mentions.append((i, key, mention))
                # since mentions are ordered, the first mention
                # outside of the blocks ends the search inside this block
                if mention.start_idx > block_end:
                    break
            # assign mentions in this co-occurrences blocks to C
            for m1, m2 in it.combinations(block_mentions, 2):
                i1, key1, mention1 = m1
                i2, key2, mention2 = m2
                # ignore co-occurrence with self
                if key1 == key2:
                    continue
                C[i1][i2] = 1

        # * Construct graph from co-occurence matrix
        G = nx.Graph()
        for character, _ in mentions:
            G.add_node(character)

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
        mentions: List[Tuple[Any, NEREntity]],
        window: Optional[int],
        overlap: int,
        dynamic_blocks_tokens: Optional[List[List[str]]],
        sentences: List[List[str]],
        sentences_polarities: Optional[List[float]],
        co_occurrences_blocks: Optional[List[Tuple[int, int]]],
    ) -> List[nx.Graph]:
        """
        .. note::

            only one of ``window`` or ``dynamic_blocks_tokens`` should be specified

        :param mentions: A list of entity mentions, ordered by
            appearance, each of the form (KEY MENTION).  KEY
            determines the unicity of the entity.
        :param window: dynamic window, in tokens.
        :param overlap: window overlap
        :param dynamic_blocks_tokens: list of tokens for each block.  If
            given, one graph will be extracted per block.
        :param co_occurrences_blocks:
        """
        assert window is None or dynamic_blocks_tokens is None
        compute_polarity = not sentences is None and not sentences_polarities is None

        if not window is None:
            return [
                self._extract_graph(
                    [elt for elt in ct if not elt is None],
                    sentences,
                    sentences_polarities,
                    co_occurrences_blocks,
                )
                for ct in windowed(mentions, window, step=window - overlap)
            ]

        assert not dynamic_blocks_tokens is None

        graphs = []

        blocks_mentions = mentions_for_blocks(dynamic_blocks_tokens, mentions)
        for block_i, (block_tokens, block_mentions) in enumerate(
            zip(dynamic_blocks_tokens, blocks_mentions)
        ):
            block_start = sum(
                [len(c) for i, c in enumerate(dynamic_blocks_tokens) if i < block_i]
            )
            block_end = block_start + len(block_tokens)
            # make mentions coordinates block local
            block_mentions = [(c, m.shifted(-block_start)) for c, m in block_mentions]

            sent_start_idx, sent_end_idx = sent_indices_for_block(
                dynamic_blocks_tokens, block_i, sentences
            )
            block_sentences = sentences[sent_start_idx : sent_end_idx + 1]

            block_sentences_polarities = None
            if compute_polarity:
                assert not sentences_polarities is None
                block_sentences_polarities = sentences_polarities[
                    sent_start_idx : sent_end_idx + 1
                ]

            if co_occurrences_blocks is None:
                block_co_occurrences_blocks = None
            else:
                block_co_occurrences_blocks = [
                    (start, end)
                    for start, end in co_occurrences_blocks
                    if start >= block_start and end <= block_end
                ]

            graphs.append(
                self._extract_graph(
                    block_mentions,
                    block_sentences,
                    block_sentences_polarities,
                    block_co_occurrences_blocks,
                )
            )

        return graphs

    def supported_langs(self) -> Union[Set[str], Literal["any"]]:
        return "any"

    def needs(self) -> Set[str]:
        needs = {"characters", "sentences"}
        if self.need_dynamic_blocks:
            needs.add("dynamic_blocks_tokens")
        if len(self.additional_ner_classes) > 0:
            needs.add("entities")
        if self.co_occurrences_dist is None:
            needs.add("co_occurrences_blocks")
        return needs

    def production(self) -> Set[str]:
        return {"character_network"}

    def optional_needs(self) -> Set[str]:
        return {"sentences_polarities"}


class ConversationalGraphExtractor(PipelineStep):
    """A graph extractor using conversation between characters

    .. note::

        This is an early version, that only supports static graphs
        for now.
    """

    def __init__(
        self, conversation_dist: Union[int, Tuple[int, Literal["tokens", "sentences"]]]
    ):
        if isinstance(conversation_dist, int):
            conversation_dist = (conversation_dist, "tokens")
        self.conversation_dist = conversation_dist

        super().__init__()

    def _quotes_interact(
        self, quote_1: Quote, quote_2: Quote, sentences: List[List[str]]
    ) -> bool:
        ordered = quote_2.start >= quote_1.end
        if self.conversation_dist[1] == "tokens":
            return (
                abs(
                    quote_2.start - quote_1.end
                    if ordered
                    else quote_1.start - quote_2.end
                )
                <= self.conversation_dist[0]
            )
        elif self.conversation_dist[1] == "sentences":
            if ordered:
                quote_1_sent = sent_index_for_token_index(quote_1.end, sentences)
                quote_2_sent = sent_index_for_token_index(quote_2.start, sentences)
            else:
                quote_1_sent = sent_index_for_token_index(quote_1.start, sentences)
                quote_2_sent = sent_index_for_token_index(quote_2.end, sentences)
            return abs(quote_1_sent - quote_2_sent) <= self.conversation_dist[0]
        else:
            raise NotImplementedError

    def __call__(
        self,
        sentences: List[List[str]],
        quotes: List[Quote],
        speakers: List[Optional[Character]],
        characters: Set[Character],
        **kwargs,
    ) -> Dict[str, Any]:
        G = nx.Graph()
        for character in characters:
            G.add_node(character)

        for i, (quote_1, speaker_1) in enumerate(zip(quotes, speakers)):
            # no speaker prediction: ignore
            if speaker_1 is None:
                continue

            # check ahead for co-occurences
            for quote_2, speaker_2 in zip(quotes[i + 1 :], speakers[i + 1 :]):
                # no speaker prediction: ignore
                if speaker_2 is None:
                    continue

                if not self._quotes_interact(quote_1, quote_2, sentences):
                    # dist between quote_1 and quote_2 is too great :
                    # we finished co-occurences search for quote_1
                    break

                # ignore co-occurences with self
                if quote_1 == quote_2 or speaker_1 == speaker_2:
                    continue

                # record co_occurence
                if not G.has_edge(speaker_1, speaker_2):
                    G.add_edge(speaker_1, speaker_2, weight=0)
                G.edges[speaker_1, speaker_2]["weight"] += 1

        return {"character_network": G}

    def needs(self) -> Set[str]:
        """sentences, quotes, speakers, characters"""
        return {"sentences", "quotes", "speakers", "characters"}

    def production(self) -> Set[str]:
        """character_network"""
        return {"character_network"}
