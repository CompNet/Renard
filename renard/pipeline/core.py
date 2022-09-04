from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Tuple, Set, List, Optional, Union, TYPE_CHECKING

from tqdm import tqdm
from transformers.tokenization_utils_base import BatchEncoding
import networkx as nx

if TYPE_CHECKING:
    from renard.pipeline.characters_extraction import Character
    from renard.pipeline.corefs.bert_corefs import CoreferenceMention


class PipelineStep:
    """An abstract pipeline step

    .. note::

        The ``__call__``, ``needs`` and ``production`` methods _must_ be
        overriden by derived classes.

    .. note::

        The ``optional_needs`` method can be overriden by derived classes.

    """

    def __init__(self) -> None:
        self.progress_report = "tqdm"

    def __call__(self, text: str, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError()

    def needs(self) -> Set[str]:
        """
        :return: a `set` of state attributes needed by this
            :class:`PipelineStep`. This method must be overriden
            by derived classes.
        """
        raise NotImplementedError()

    def optional_needs(self) -> Set[str]:
        """
        :return: a `set` of state attributes optionally neeeded by this
            :class:`PipelineStep`. This method can be overriden by derived
            classes.
        """
        return set()

    def production(self) -> Set[str]:
        """
        :return: a `set` of state attributes produced by this
            :class:`PipelineStep`. This method must be overriden
            by derived classes.
        """
        raise NotImplementedError()


@dataclass
class PipelineState:
    """The state of a pipeline, annotated in a :class:`Pipeline` lifetime"""

    #: input text
    text: str

    #: text splitted in tokens
    tokens: Optional[List[str]] = None
    #: word piece tokens, for BERT-like models
    wp_tokens: Optional[List[str]] = None

    #: BIO NER tags, aligned with ``self.tokens``
    bio_tags: Optional[List[str]] = None
    #: BIO NER tags, aligned with ``self.wp_tokens``
    wp_bio_tags: Optional[List[str]] = None
    #: BERT batch encodings
    bert_batch_encoding: Optional[BatchEncoding] = None

    #: coreference chains
    corefs: Optional[List[List[CoreferenceMention]]] = None

    #: detected characters
    characters: Optional[Set[Character]] = None

    #: characters graph
    characters_graph: Optional[Union[List[nx.Graph], nx.Graph]] = None

    def export_graph_to_gexf(self, path: str):
        """Export characters graph to Gephi's gexf format

        :param path: export file path
        """
        if not isinstance(self.characters_graph, nx.Graph):
            raise RuntimeError(
                f"characters graph cant be exported : {self.characters_graph}"
            )
        G = nx.relabel_nodes(
            self.characters_graph,
            {
                character: character.longest_name()  # type: ignore
                for character in self.characters_graph.nodes()
            },
        )
        nx.write_gexf(G, path)


class Pipeline:
    """A flexible NLP pipeline"""

    def __init__(
        self, steps: Tuple[PipelineStep, ...], progress_report: Optional[str] = "tqdm"
    ) -> None:
        """
        :param steps: a ``tuple`` of :class:``PipelineStep``, that will be executed in order
        :param progress_report: if ``tqdm``, report the pipeline progress using tqdm. Otherwise,
            does not report progress. This sets the ``progress_report`` attribute for all steps.
        """
        self.steps = steps
        self.progress_report = progress_report
        for step in self.steps:
            step.progress_report = progress_report

    def check_valid(self) -> Tuple[bool, List[str]]:
        """Check that the current pipeline can be run, which is
        possible if all steps needs are satisfied

        :return: a tuple : ``(True, [warnings])`` if the pipeline is valid,
            ``(False, [errors])`` otherwise
        """

        pipeline_state = {"text"}
        warnings = []

        for i, step in enumerate(self.steps):

            if not step.needs().issubset(pipeline_state):
                return (
                    False,
                    [
                        f"step {i + 1} ({step.__class__.__name__}) has unsatisfied needs (needs : {step.needs()}, available : {pipeline_state})"
                    ],
                )

            if not step.optional_needs().issubset(pipeline_state):
                warnings.append(
                    f"step {i + 1} ({step.__class__.__name__}) has unsatisfied optional needs : (optional needs : {step.optional_needs()}, available : {pipeline_state})"
                )

            pipeline_state = pipeline_state.union(step.production())

        return (True, warnings)

    def __call__(self, text: str) -> PipelineState:
        """Run the pipeline sequentially

        :return: the output of the last step of the pipeline
        """
        is_valid, warnings_or_errors = self.check_valid()
        if not is_valid:
            raise ValueError(warnings_or_errors)
        for warning in warnings_or_errors:
            print(f"[warning] : {warning}")

        state = PipelineState(text)

        if self.progress_report == "tqdm":
            steps = tqdm(self.steps, total=len(self.steps))
        else:
            steps = self.steps

        for step in steps:

            if isinstance(steps, tqdm):
                steps.set_description_str(f"{step.__class__.__name__}")

            out = step(**state.__dict__)
            for key, value in out.items():
                setattr(state, key, value)

        return state
