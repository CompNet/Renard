import itertools
from typing import Dict, Any, List, Set, Optional, Tuple, Literal, Union
import operator
from itertools import accumulate

import networkx as nx
import numpy as np
from more_itertools import windowed

from renard.pipeline.ner import NEREntity
from renard.pipeline.core import PipelineStep
from renard.pipeline.character_unification import Character
from renard.pipeline.quote_detection import Quote


def sent_index_for_token_index(token_index: int, sentences: List[List[str]]) -> int:
    """Compute the index of the sentence of the token at ``token_index``"""
    sents_len = accumulate([len(s) for s in sentences], operator.add)
    return next((i for i, l in enumerate(sents_len) if l > token_index))


def sent_indices_for_blocks(
    dynamic_blocks: List[List[str]], block_idx: int, sentences: List[List[str]]
) -> Tuple[int, int]:
    """Return the indices of the first and the last sentence of a
    block

    :param dynamic_blocks: all blocks
    :param block_idx: index of the block for which sentence
        indices are returned
    :param sentences: all sentences
    :return: ``(first sentence index, last sentence index)``
    """
    block_start_idx = sum(
        [len(c) for i, c in enumerate(dynamic_blocks) if i < block_idx]
    )
    block_end_idx = block_start_idx + len(dynamic_blocks[block_idx])
    sents_start_idx = None
    sents_end_idx = None
    count = 0
    for sent_i, sent in enumerate(sentences):
        start_idx, end_idx = (count, count + len(sent))
        count = end_idx
        if sents_start_idx is None and start_idx >= block_start_idx:
            sents_start_idx = sent_i
        if sents_end_idx is None and end_idx >= block_end_idx:
            sents_end_idx = sent_i
            break
    assert not sents_start_idx is None and not sents_end_idx is None
    return (sents_start_idx, sents_end_idx)


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
        itertools.accumulate([0] + [len(block) for block in dynamic_blocks[:-1]])
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
        ],
        dynamic: bool = False,
        dynamic_window: Optional[int] = None,
        dynamic_overlap: int = 0,
        co_occurences_dist: Optional[
            Union[int, Tuple[int, Literal["tokens", "sentences"]]]
        ] = None,
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

        :param co_occurences_dist: same as ``co_occurrences_dist``.
            Included because of retro-compatibility, as it was a
            previously included typo.

        :param additional_ner_classes: if specified, will include
            entities other than characters in the final graph.  No
            attempt will be made at unfying the entities (for example,
            "New York" will be distinct from "New York City").
        """
        # typo retrocompatibility
        if not co_occurences_dist is None:
            co_occurrences_dist = co_occurences_dist
        if co_occurrences_dist is None and co_occurences_dist is None:
            raise ValueError()

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
        **kwargs,
    ) -> Dict[str, Any]:
        """Extract a characters graph

        :param characters:

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
                )
            }
        return {
            "character_network": self._extract_graph(
                mentions, sentences, sentences_polarities
            )
        }

    def _mentions_interact(
        self,
        mention_1: NEREntity,
        mention_2: NEREntity,
        sentences: Optional[List[List[str]]] = None,
    ) -> bool:
        """Check if two mentions are close enough to be in interactions.

        .. note::

            the attribute ``self.co_occurrences_dist`` is used to know wether mentions are in co_occurences

        :param mention_1:
        :param mention_2:
        :param sentences:
        :return: a boolean indicating wether the two mentions are co-occuring
        """
        assert not self.co_occurrences_dist is None
        if self.co_occurrences_dist[1] == "tokens":
            return (
                abs(mention_2.start_idx - mention_1.start_idx)
                <= self.co_occurrences_dist[0]
            )
        elif self.co_occurrences_dist[1] == "sentences":
            assert not sentences is None
            mention_1_sent = sent_index_for_token_index(mention_1.start_idx, sentences)
            mention_2_sent = sent_index_for_token_index(
                mention_2.end_idx - 1, sentences
            )
            return abs(mention_2_sent - mention_1_sent) <= self.co_occurrences_dist[0]
        else:
            raise NotImplementedError

    def _extract_graph(
        self,
        mentions: List[Tuple[Any, NEREntity]],
        sentences: List[List[str]],
        sentences_polarities: Optional[List[float]],
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

        # co-occurence matrix, where C[i][j] is 1 when appearance
        # i co-occur with j if i < j, or 0 when it doesn't
        C = np.zeros((len(mentions), len(mentions)))
        for i, (char1, mention_1) in enumerate(mentions):
            # check ahead for co-occurences
            for j, (char2, mention_2) in enumerate(mentions[i + 1 :]):
                if not self._mentions_interact(mention_1, mention_2, sentences):
                    # dist between current token and future token is
                    # too great : we finished co-occurences search for
                    # the current token
                    break
                # ignore co-occurences with self
                if char1 == char2:
                    continue
                # record co_occurence
                C[i][i + 1 + j] = 1

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
        """
        assert window is None or dynamic_blocks_tokens is None
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

        assert not dynamic_blocks_tokens is None

        graphs = []

        blocks_mentions = mentions_for_blocks(dynamic_blocks_tokens, mentions)
        for block_i, (_, block_mentions) in enumerate(
            zip(dynamic_blocks_tokens, blocks_mentions)
        ):
            block_start_idx = sum(
                [len(c) for i, c in enumerate(dynamic_blocks_tokens) if i < block_i]
            )
            # make mentions coordinates block local
            block_mentions = [
                (c, m.shifted(-block_start_idx)) for c, m in block_mentions
            ]

            sent_start_idx, sent_end_idx = sent_indices_for_blocks(
                dynamic_blocks_tokens, block_i, sentences
            )
            block_sentences = sentences[sent_start_idx : sent_end_idx + 1]

            block_sentences_polarities = None
            if compute_polarity:
                assert not sentences_polarities is None
                block_sentences_polarities = sentences_polarities[
                    sent_start_idx : sent_end_idx + 1
                ]

            graphs.append(
                self._extract_graph(
                    block_mentions,
                    block_sentences,
                    block_sentences_polarities,
                )
            )

        return graphs

    def _extract_gephi_dynamic_graph(
        self, mentions: List[Tuple[Character, NEREntity]], sentences: List[List[str]]
    ) -> nx.Graph:
        """
        :param mentions: A list of character mentions, ordered by appearance
        :param sentences:
        """
        # keep only longest name in graph node : possible only if it is unique
        # TODO: might want to try and get shorter names if longest names aren't
        #       unique
        characters = set([e[0] for e in mentions])

        G = nx.Graph()

        character_to_last_appearance: Dict[Character, Optional[NEREntity]] = {
            character: None for character in characters
        }

        for i, (character, mention) in enumerate(mentions):
            if not character in characters:
                continue
            character_to_last_appearance[character] = mention
            close_characters = [
                c
                for c, last_appearance in character_to_last_appearance.items()
                if not last_appearance is None
                and self._mentions_interact(mention, last_appearance, sentences)
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

    def supported_langs(self) -> Union[Set[str], Literal["any"]]:
        return "any"

    def needs(self) -> Set[str]:
        needs = {"characters", "sentences"}
        if self.need_dynamic_blocks:
            needs.add("dynamic_blocks_tokens")
        if len(self.additional_ner_classes) > 0:
            needs.add("entities")
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
