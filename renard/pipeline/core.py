from dataclasses import dataclass
from typing import Any, Dict, Tuple, Set, List, Optional, Union

from tqdm import tqdm
from transformers.tokenization_utils_base import BatchEncoding
import networkx as nx


class PipelineStep:
    """An abstract pipeline step, executed sequentially

    .. note::

        The ``__call__``, ``needs`` and ``production`` methods _must_ be
        overriden by derived classes.

    .. note::

        The ``optional_needs`` method can be overriden by derived classes.

    """

    def __call__(self, text: str, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError()

    def needs(self) -> Set[str]:
        raise NotImplementedError()

    def optional_needs(self) -> Set[str]:
        return set()

    def production(self) -> Set[str]:
        raise NotImplementedError()


@dataclass
class PipelineState:
    """The state of a pipeline, annotated in a Pipeline lifetime"""

    text: str

    # tokenization
    tokens: Optional[List[str]] = None
    ## BERT tokenization
    wp_tokens: Optional[List[str]] = None

    # NER
    bio_tags: Optional[List[str]] = None
    ## BERT NER
    wp_bio_tags: Optional[List[str]] = None
    bert_batch_encoding: Optional[BatchEncoding] = None

    # coreference resolution
    corefs: Optional[List[List[dict]]] = None

    # character detection
    characters: Optional[Set["Character"]] = None

    # graph extraction
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

    def __init__(self, steps: Tuple[PipelineStep, ...]) -> None:
        self.steps = steps

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
        tqdm_steps = tqdm(self.steps, total=len(self.steps))
        for step in tqdm_steps:
            tqdm_steps.set_description_str(f"{step.__class__.__name__}")
            out = step(**state.__dict__)
            for key, value in out.items():
                setattr(state, key, value)

        return state
