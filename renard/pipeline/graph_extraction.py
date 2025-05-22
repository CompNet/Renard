from typing import Dict, Any, List, Set, Optional, Tuple, Literal, Union
import itertools as it
import operator

import networkx as nx
import numpy as np
from more_itertools import windowed

from renard.utils import BlockBounds, charbb2tokenbb
from renard.pipeline.ner import NEREntity
from renard.pipeline.core import PipelineStep
from renard.pipeline.character_unification import Character
from renard.pipeline.quote_detection import Quote


def sent_index_for_token_index(token_index: int, sentences: List[List[str]]) -> int:
    """Compute the index of the sentence of the token at ``token_index``"""
    sents_len = it.accumulate([len(s) for s in sentences], operator.add)
    return next((i for i, l in enumerate(sents_len) if l > token_index))


def sent_indices_for_block(
    dynamic_block: Tuple[int, int], sentences: List[List[str]]
) -> Tuple[int, int]:
    """Return the indices of the first and the last sentence of a
    block

    :param dynamic_block: (START, END) in tokens
    :return: ``(first sentence index, last sentence index)``
    """
    block_start, block_end = dynamic_block
    sents_start = None
    sents_end = None
    count = 0
    for sent_i, sent in enumerate(sentences):
        start, end = (count, count + len(sent))
        count = end
        if sents_start is None and start >= block_start:
            sents_start = sent_i
        if sents_end is None and end >= block_end:
            # this happens when the block is _smaller_ than the
            # current sentence. In that case, we return the current
            # sentence even though it overflows the block.
            if sents_start is None:
                sents_start = sent_i
            sents_end = sent_i
            break
    assert not sents_start is None and not sents_end is None
    return (sents_start, sents_end)


def mentions_for_blocks(
    block_bounds: BlockBounds,
    mentions: List[Tuple[Any, NEREntity]],
) -> List[List[Tuple[Any, NEREntity]]]:
    """Return each block mentions.

    :param block_bounds: block bounds, in tokens
    :param mentions:

    :return: a list of mentions per blocks.  This list has len
             ``len(block_bounds)``.
    """
    assert block_bounds[1] == "tokens"

    blocks_mentions = [[] for _ in range(len(block_bounds[0]))]

    for mention in mentions:
        for block_i, (start_i, end_i) in enumerate(block_bounds[0]):
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
        self.need_co_occurrences_blocks = co_occurrences_dist is None

        if dynamic:
            if not dynamic_window is None:
                assert dynamic_window > 0
                assert dynamic_window > dynamic_overlap
        self.dynamic = dynamic
        self.dynamic_window = dynamic_window
        self.dynamic_overlap = dynamic_overlap
        self.need_dynamic_blocks = dynamic and dynamic_window is None

        self.additional_ner_classes = additional_ner_classes or []

        super().__init__()

    def __call__(
        self,
        characters: Set[Character],
        sentences: List[List[str]],
        char2token: Optional[List[int]] = None,
        dynamic_blocks: Optional[BlockBounds] = None,
        sentences_polarities: Optional[List[float]] = None,
        entities: Optional[List[NEREntity]] = None,
        co_occurrences_blocks: Optional[BlockBounds] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Extract a co-occurrence character network.

        :param co_occurrences_blocks: custom blocks where
            co-occurrences should be recorded.  For example, this can
            be used to perform chapter level co-occurrences.

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

        # convert from char blocks to token blocks
        if not dynamic_blocks is None and dynamic_blocks[1] == "characters":
            assert not char2token is None
            dynamic_blocks = charbb2tokenbb(dynamic_blocks, char2token)
        if (
            not co_occurrences_blocks is None
            and co_occurrences_blocks[1] == "characters"
        ):
            assert not char2token is None
            co_occurrences_blocks = charbb2tokenbb(co_occurrences_blocks, char2token)

        if self.dynamic:
            return {
                "character_network": self._extract_dynamic_graph(
                    mentions,
                    self.dynamic_window,
                    self.dynamic_overlap,
                    dynamic_blocks,
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
    ) -> BlockBounds:
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
            return (blocks, "tokens")

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
            blocks = [
                (start, end)
                for start, end in sorted(blocks_indices, key=lambda indices: indices[0])
            ]
            return (blocks, "tokens")

        else:
            raise ValueError(
                f"co_occurrences_dist unit should be one of: 'tokens', 'sentences'"
            )

    def _extract_graph(
        self,
        mentions: List[Tuple[Any, NEREntity]],
        sentences: List[List[str]],
        sentences_polarities: Optional[List[float]],
        co_occurrences_blocks: Optional[BlockBounds],
    ) -> nx.Graph:
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
        :param co_occurrences_blocks: only unit 'tokens' is accepted.
        """
        compute_polarity = not sentences_polarities is None

        assert co_occurrences_blocks is None or co_occurrences_blocks[1] == "tokens"
        if co_occurrences_blocks is None:
            co_occurrences_blocks = self._create_co_occurrences_blocks(
                sentences, mentions
            )

        # co-occurence matrix, where C[i][j] is 1 when appearance
        # i co-occur with j if i < j, or 0 when it doesn't
        C = np.zeros((len(mentions), len(mentions)))
        for block_start, block_end in co_occurrences_blocks[0]:
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
        for character, mention in mentions:
            # NOTE: we add an 'entity_type' attribute. This is useful
            # when using the 'additional_ner_classes' option, to
            # differentiate between different entity types.
            G.add_node(character, entity_type=mention.tag)

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
        dynamic_blocks: Optional[BlockBounds],
        sentences: List[List[str]],
        sentences_polarities: Optional[List[float]],
        co_occurrences_blocks: Optional[BlockBounds],
    ) -> List[nx.Graph]:
        """
        .. note::

            only one of ``window`` or ``dynamic_blocks_tokens`` should be specified

        :param mentions: A list of entity mentions, ordered by
            appearance, each of the form (KEY MENTION).  KEY
            determines the unicity of the entity.
        :param window: dynamic window, in tokens.
        :param overlap: window overlap
        :param dynamic_blocks: boundaries of each dynamic block
        :param co_occurrences_blocks: boundaries of each co-occurrences blocks
        """
        assert co_occurrences_blocks is None or co_occurrences_blocks[1] == "tokens"
        assert window is None or dynamic_blocks is None
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

        assert not dynamic_blocks is None

        graphs = []

        blocks_mentions = mentions_for_blocks(dynamic_blocks, mentions)
        for dynamic_block, block_mentions in zip(dynamic_blocks[0], blocks_mentions):
            block_start, block_end = dynamic_block

            sent_start, sent_end = sent_indices_for_block(dynamic_block, sentences)
            block_sentences = sentences[sent_start : sent_end + 1]

            block_sentences_polarities = None
            if compute_polarity:
                assert not sentences_polarities is None
                block_sentences_polarities = sentences_polarities[
                    sent_start : sent_end + 1
                ]

            if co_occurrences_blocks is None:
                block_co_occ_bounds = None
            else:
                bounds = [
                    (start, end)
                    for start, end in co_occurrences_blocks[0]
                    if start >= block_start and end <= block_end
                ]
                block_co_occ_bounds = (bounds, "tokens")

            graphs.append(
                self._extract_graph(
                    block_mentions,
                    block_sentences,
                    block_sentences_polarities,
                    block_co_occ_bounds,
                )
            )

        return graphs

    def supported_langs(self) -> Union[Set[str], Literal["any"]]:
        return "any"

    def needs(self) -> Set[str]:
        needs = {"characters", "sentences"}

        if self.need_dynamic_blocks:
            needs.add("dynamic_blocks")
            needs.add("char2token")
        if self.need_co_occurrences_blocks:
            needs.add("co_occurrences_blocks")
            needs.add("char2token")

        if len(self.additional_ner_classes) > 0:
            needs.add("entities")

        return needs

    def production(self) -> Set[str]:
        return {"character_network"}

    def optional_needs(self) -> Set[str]:
        return {"sentences_polarities"}


class ConversationalGraphExtractor(PipelineStep):
    """A graph extractor using conversation between characters or
    mentions.

    .. note::

        Does not support dynamic networks yet.
    """

    def __init__(
        self,
        graph_type: Literal["conversation", "mention"],
        conversation_dist: Optional[
            Union[int, Tuple[int, Literal["tokens", "sentences"]]]
        ] = None,
        ignore_self_mention: bool = True,
    ):
        """
        :param graph_type: either 'conversation' or 'mention'.
            'conversation' extracts an undirected graph with
            interactions being extracted from the conversations
            occurring between characters.  'mention' extracts a
            directed graph where interactions are character mentions
            of one another in quoted speech.
        :param conversation_dist: must be supplied if `graph_type` is
            'conversation'.  The distance between two quotation for
            them to be considered as being interacting.
        :param ignore_self_mention: if ``True``, self mentions are
            ignore for ``graph_type=='mention'``
        """
        self.graph_type = graph_type

        if isinstance(conversation_dist, int):
            conversation_dist = (conversation_dist, "tokens")
        self.conversation_dist = conversation_dist

        self.ignore_self_mention = ignore_self_mention

        super().__init__()

    def _quotes_interact(
        self, quote_1: Quote, quote_2: Quote, sentences: List[List[str]]
    ) -> bool:
        assert not self.conversation_dist is None
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

    def _conversation_extract(
        self,
        sentences: List[List[str]],
        quotes: List[Quote],
        speakers: List[Optional[Character]],
        characters: Set[Character],
    ) -> nx.Graph:
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

        return G

    def _mention_extract(
        self,
        quotes: List[Quote],
        speakers: List[Optional[Character]],
        characters: Set[Character],
    ) -> nx.Graph:
        G = nx.DiGraph()
        for character in characters:
            G.add_node(character)

        for quote, speaker in zip(quotes, speakers):
            # no speaker prediction: ignore
            if speaker is None:
                continue

            # TODO: optim
            # find characters mentioned in quote and add a directed
            # edge speaker => character
            for character in characters:
                if character == speaker and self.ignore_self_mention:
                    continue
                for mention in character.mentions:
                    if (
                        mention.start_idx >= quote.start
                        and mention.end_idx <= quote.end
                    ):
                        if not G.has_edge(speaker, character):
                            G.add_edge(speaker, character, weight=0)
                        G.edges[speaker, character]["weight"] += 1
                        break

        return G

    def __call__(
        self,
        sentences: List[List[str]],
        quotes: List[Quote],
        speakers: List[Optional[Character]],
        characters: Set[Character],
        **kwargs,
    ) -> Dict[str, Any]:

        if self.graph_type == "conversation":
            G = self._conversation_extract(sentences, quotes, speakers, characters)
        elif self.graph_type == "mention":
            G = self._mention_extract(quotes, speakers, characters)
        else:
            raise ValueError(f"unknown graph_type: {self.graph_type}")

        return {"character_network": G}

    def needs(self) -> Set[str]:
        """sentences, quotes, speakers, characters"""
        return {"sentences", "quotes", "speakers", "characters"}

    def production(self) -> Set[str]:
        """character_network"""
        return {"character_network"}
