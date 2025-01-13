import os
from typing import List, Optional, Set, Dict, Any, Literal
from renard.ner_utils import ner_entities

import stanza
from stanza.protobuf import CoreNLP_pb2
from stanza.server import CoreNLPClient
from stanza.resources.installation import DEFAULT_CORENLP_DIR

from renard.pipeline.core import PipelineStep, Mention


def corenlp_is_installed() -> bool:
    return os.path.exists(DEFAULT_CORENLP_DIR)


def corenlp_annotations_bio_tags(annotations: CoreNLP_pb2.Document) -> List[str]:
    """Returns an array of bio tags extracted from stanford corenlp annotations

    .. note::

        only PERSON, LOCATION, ORGANIZATION and MISC entities are considered.
        Other types of entities are discarded.
        (see https://stanfordnlp.github.io/CoreNLP/ner.html#description) for
        a list of usual coreNLP types.

    .. note::

        Weirdly, CoreNLP will annotate pronouns as entities. Only tokens having
        a NNP POS are kept by this function.

    :param annotations: stanford coreNLP text annotations
    :return: an array of bio tags.
    """
    corenlp_tokens = [
        token for sentence in annotations.sentence for token in sentence.token  # type: ignore
    ]
    bio_tags = ["O"] * len(corenlp_tokens)

    stanford_to_bio = {
        "PERSON": "PER",
        "LOCATION": "LOC",
        "ORGANIZATION": "ORG",
        "MISC": "MISC",
    }

    for mention in annotations.mentions:  # type: ignore

        # ignore tags not in conll 2003 format
        if not mention.ner in stanford_to_bio:
            continue

        token_start_idx = mention.tokenStartInSentenceInclusive
        token_end_idx = mention.tokenEndInSentenceExclusive

        # ignore entities having a pos different than NNP
        if corenlp_tokens[token_start_idx].pos != "NNP":
            continue

        bio_tag = f"B-{stanford_to_bio[mention.ner]}"
        bio_tags[token_start_idx] = bio_tag
        for i in range(token_start_idx + 1, token_end_idx):
            bio_tag = f"I-{stanford_to_bio[mention.ner]}"
            bio_tags[i] = bio_tag

    return bio_tags


class StanfordCoreNLPPipeline(PipelineStep):
    """a full NLP pipeline using stanford CoreNLP

    .. note::

        The Stanford CoreNLP pipeline requires the ``stanza`` library.
        You can install it with poetry using ``pip install stanza``.

    .. warning::

        RAM usage might be high for coreference resolutions as it uses
        the entire novel ! If CoreNLP terminates with an out of memory
        error, you can try allocating more memory for the server by
        using ``server_kwargs`` (example : ``{"memory": "8G"}``).
    """

    def __init__(
        self,
        annotate_corefs: bool = False,
        corefs_algorithm: Literal[
            "deterministic", "statistical", "neural"
        ] = "statistical",
        corenlp_custom_properties: Optional[Dict[str, Any]] = None,
        server_timeout: int = 9999999,
        **server_kwargs,
    ) -> None:
        """
        :param annotate_corefs: ``True`` if coreferences must be
            annotated, ``False`` otherwise. This parameter is not
            yet implemented.

        :param corefs_algorithm: one of ``{"deterministic", "statistical", "neural"}``

        :param corenlp_custom_properties: custom properties dictionary to pass to the
            CoreNLP server. Note that some properties are already set when calling the
            server, so not all properties are supported : it is intended as a last
            resort escape hatch. In particular, do not set ``'ner.applyFineGrained'``.
            If you need to set the coreference algorithm used, see ``corefs_algorithm``.

        :param server_timeout: CoreNLP server timeout in ms

        :param server_kwargs: extra args for stanford CoreNLP server. `be_quiet`
            and `max_char_length` are *not* supported.
            See here for a list of possible args :
            https://stanfordnlp.github.io/stanza/client_properties.html#corenlp-server-start-options-server
        """
        assert corefs_algorithm in {"deterministic", "statistical", "neural"}
        self.annotate_corefs = annotate_corefs
        self.corefs_algorithm = corefs_algorithm

        self.server_timeout = server_timeout
        self.server_kwargs = server_kwargs
        self.corenlp_custom_properties = (
            corenlp_custom_properties if not corenlp_custom_properties is None else {}
        )
        super().__init__()

    def __call__(self, text: str, **kwargs) -> Dict[str, Any]:
        if not corenlp_is_installed():
            stanza.install_corenlp()

        # define corenlp annotators and properties
        corenlp_annotators = ["tokenize", "ssplit", "pos", "lemma", "ner"]
        corenlp_properties = {
            **self.corenlp_custom_properties,
            **{"ner.applyFineGrained": False},
        }

        ## coreference annotation settings
        if self.annotate_corefs:
            if self.corefs_algorithm == "deterministic":
                corenlp_annotators += ["parse", "dcoref"]
            elif self.corefs_algorithm == "statistical":
                corenlp_annotators += ["depparse", "coref"]
                corenlp_properties = {
                    **corenlp_properties,
                    **{"coref.algorithm": "statistical"},
                }
            elif self.corefs_algorithm == "neural":
                corenlp_annotators += ["depparse", "coref"]
                corenlp_properties = {
                    **corenlp_properties,
                    **{"coref.algorithm": "neural"},
                }
            else:
                raise RuntimeError(
                    f"unknown coref algorithm : {self.corefs_algorithm}."
                )

        with CoreNLPClient(
            annotators=corenlp_annotators,
            max_char_length=len(text),
            timeout=self.server_timeout,
            be_quiet=True,
            properties=corenlp_properties,
            **self.server_kwargs,
        ) as client:

            # compute annotation
            annotations: CoreNLP_pb2.Document = client.annotate(text)  # type: ignore

            # parse tokens
            tokens = [
                token.word
                for sentence in annotations.sentence  # type: ignore
                for token in sentence.token
            ]

            # parse NER bio tags
            bio_tags = corenlp_annotations_bio_tags(annotations)

            # parse corefs if enabled
            if self.annotate_corefs:

                coref_chains = []

                for coref_chain in annotations.corefChain:  # type: ignore

                    chain = []

                    for mention in coref_chain.mention:  # type: ignore

                        mention_sent = annotations.sentence[mention.sentenceIndex]  # type: ignore
                        sent_start_idx = mention_sent.token[0].tokenBeginIndex

                        mention_words = []
                        for token in mention_sent.token[
                            mention.beginIndex : mention.endIndex - 1
                        ]:
                            mention_words.append(token.word)
                            mention_words.append(token.after)
                        mention_words.append(
                            mention_sent.token[mention.endIndex - 1].word
                        )

                        chain.append(
                            Mention(
                                mention_words,
                                sent_start_idx + mention.beginIndex,
                                sent_start_idx + mention.endIndex,
                            )
                        )

                    coref_chains.append(chain)

        out_dict = {"tokens": tokens, "entities": ner_entities(tokens, bio_tags)}
        if self.annotate_corefs:
            out_dict["corefs"] = coref_chains  # type: ignore
        return out_dict

    def needs(self) -> Set[str]:
        return set()

    def production(self) -> Set[str]:
        production = {"tokens", "entities"}
        if self.annotate_corefs:
            production.add("corefs")
        return production
