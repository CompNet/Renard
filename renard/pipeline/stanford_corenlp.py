import os
from typing import List, Set, Dict, Any

from tqdm import tqdm
import stanza
from stanza.protobuf import CoreNLP_pb2
from stanza.server import CoreNLPClient
from stanza.resources.installation import DEFAULT_CORENLP_DIR

from renard.pipeline.core import PipelineStep
from renard.utils import sliding_window


def corenlp_is_installed() -> bool:
    return os.path.exists(DEFAULT_CORENLP_DIR)


def corenlp_annotations_sentences(annotations: CoreNLP_pb2.Document) -> List[str]:
    """Extract an array of sentences from stanford corenlp annotations

    :param annotations: stanford CoreNLP text annotations
    :return: an array of sentences
    """
    sentences = []
    for sentence in annotations.sentence:  # type: ignore
        current_sentence = []
        for token in sentence.token:
            current_sentence.append(token.word)
            current_sentence.append(token.after)
        sentences.append("".join(current_sentence))
    return sentences


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

        only supports english for now

    * TODO description when coref is implemented
    :ivar annotate_corefs: ``True`` if coreferences must be annotated,
        ``False`` otherwise. This parameter is not yet implemented
    """

    def __init__(self, annotate_corefs: bool = False) -> None:
        self.annotate_corefs = annotate_corefs
        # TODO remove message when coref is implemented
        if annotate_corefs:
            print("[warning] : coreference annotation is not yet supported")

    def __call__(self, text: str, **kwargs) -> Dict[str, Any]:
        if not corenlp_is_installed():
            stanza.install_corenlp()

        # 1. tokenization + ner
        corenlp_annotators = ["tokenize", "ssplit", "pos", "lemma", "ner"]
        with CoreNLPClient(
            annotators=corenlp_annotators,
            max_char_length=len(text),
            timeout=9999999,
            be_quiet=True,
            properties={"ner.applyFineGrained": False},
        ) as client:
            annotations: CoreNLP_pb2.Document = client.annotate(text)  # type: ignore
            tokens = [
                token.word
                for sentence in annotations.sentence  # type: ignore
                for token in sentence.token
            ]
            bio_tags = corenlp_annotations_bio_tags(annotations)

            # 2. corefs with sliding window on sentence
            #
            # * TODO batch requests
            #   from the stanza doc : documents can be concatenated together
            #   by separating them with two line breaks
            if self.annotate_corefs:
                sentences = corenlp_annotations_sentences(annotations)
                # * TODO n as parameter
                for context_sentences in tqdm(
                    sliding_window(sentences, n=3), total=len(sentences) / 3
                ):
                    # * TODO not only dcoref
                    #   to change the coref algorithm when using "coref" annotator :
                    #   set "coref.algorithm='neural'"
                    client.annotate(
                        " ".join(context_sentences),
                        annotators=corenlp_annotators + ["parse", "dcoref"],
                    )

        # * TODO coref annotations parsing
        return {"tokens": tokens, "bio_tags": bio_tags}

    def needs(self) -> Set[str]:
        return set()

    def produces(self) -> Set[str]:
        production = {"tokens", "bio_tags"}
        if self.annotate_corefs:
            production.add("corefs")
        return production
