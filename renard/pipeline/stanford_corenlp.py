import os
from typing import List, Set, Dict, Any
import stanza
from stanza.protobuf import CoreNLP_pb2
from stanza.server import CoreNLPClient
from stanza.resources.installation import DEFAULT_CORENLP_DIR
from renard.pipeline.core import PipelineStep


class StanfordCoreNLPPipeline(PipelineStep):
    """a full NLP pipeline using stanford CoreNLP

    .. note::

        only supports english for now

    :ivar annotate_corefs: ``True`` if coreferences must be annotated,
        ``False`` otherwise. This parameter is not yet implemented TODO corefs
    """

    def __init__(self, annotate_corefs: bool = False) -> None:
        self.annotate_corefs = annotate_corefs
        if annotate_corefs:  # TODO corefs
            print("[warning] : coreference annotation is not yet supported")

    def __call__(self, text: str, **kwargs) -> Dict[str, Any]:
        if not StanfordCoreNLPPipeline.corenlp_is_installed():
            stanza.install_corenlp()

        # annotate text using coreNLP server
        corenlp_annotators = ["tokenize", "ssplit", "pos", "lemma", "ner"]
        if self.annotate_corefs:
            corenlp_annotators.append("dcoref")
        with CoreNLPClient(
            annotators=corenlp_annotators,
            max_char_length=len(text),
            timeout=99999999,  # TODO
            be_quiet=True,
            properties={"ner.applyFineGrained": False},
        ) as client:
            annotations: CoreNLP_pb2.Document = client.annotate(text)  # type: ignore

        tokens = [
            token.word
            for sentence in annotations.sentence  # type: ignore
            for token in sentence.token
        ]
        bio_tags = StanfordCoreNLPPipeline.annotations_bio_tags(annotations)

        return {"tokens": tokens, "bio_tags": bio_tags}

    @staticmethod
    def corenlp_is_installed() -> bool:
        return os.path.exists(DEFAULT_CORENLP_DIR)

    @staticmethod
    def annotations_bio_tags(annotations: CoreNLP_pb2.Document) -> List[str]:
        """Returns an array of bio tags extracted from stanford corenlp annotation

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
            token for sentence in annotations.sentence for token in sentence.token
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

    def needs(self) -> Set[str]:
        return set()

    def produces(self) -> Set[str]:
        production = {"tokens", "bio_tags"}
        if self.annotate_corefs:
            production.add("corefs")
        return production
