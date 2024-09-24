from __future__ import annotations
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Literal,
    Iterable,
    Tuple,
    Set,
    List,
    Optional,
    Union,
    TypeVar,
    Type,
    TYPE_CHECKING,
)
import os, sys

import networkx as nx
from networkx.readwrite.gexf import GEXFWriter

from renard.pipeline.progress import ProgressReporter, get_progress_reporter, progress_
from renard.plot_utils import plot_nx_graph_reasonably, layout_nx_graph_reasonably
from renard.graph_utils import (
    cumulative_graph,
    graph_with_names,
    dynamic_graph_to_gephi_graph,
    layout_with_names,
)

if TYPE_CHECKING:
    from renard.plot_utils import CharactersGraphLayout
    from renard.pipeline.character_unification import Character
    from renard.pipeline.ner import NEREntity
    from renard.pipeline.quote_detection import Quote
    import matplotlib.pyplot as plt


@dataclass
class Mention:
    tokens: List[str]
    start_idx: int
    end_idx: int

    def shifted(self, shift: int) -> Mention:
        self_dict = vars(self)
        self_dict["start_idx"] = self.start_idx + shift
        self_dict["end_idx"] = self.end_idx + shift
        return self.__class__(**self_dict)

    def __eq__(self, other: Mention) -> bool:
        return (
            self.tokens == other.tokens
            and self.start_idx == other.start_idx
            and self.end_idx == other.end_idx
        )

    def __hash__(self) -> int:
        return hash(tuple(self.tokens) + (self.start_idx, self.end_idx))


class PipelineStep:
    """An abstract pipeline step

    .. note::

        The ``__call__``, ``needs`` and ``production`` methods _must_ be
        overridden by derived classes.

    .. note::

        The ``optional_needs`` and ``supported_langs`` methods can be
        overridden by derived classes.
    """

    def __init__(self):
        """Initialize the :class:`PipelineStep` with a given configuration."""
        pass

    def _pipeline_init_(
        self, lang: str, progress_reporter: ProgressReporter, **kwargs
    ) -> Optional[Dict[Pipeline.PipelineParameter, Any]]:
        """Set the step configuration that is common to the whole
        pipeline.

        :param lang: the lang of the whole pipeline
        :param progress_reporter:
        :param kwargs: additional pipeline parameters.

        :return: a step can return a dictionary of pipeline params if
                 it wish to modify some of these.
        """
        supported_langs = self.supported_langs()
        if not supported_langs == "any" and not lang in supported_langs:
            raise ValueError(
                f"[error] {self.__class__} does not support lang {lang} (supported language: {supported_langs})."
            )
        self.lang = lang

        self.progress_reporter = progress_reporter

    T = TypeVar("T")

    def _progress_(
        self, it: Iterable[T], total: Optional[int] = None
    ) -> Generator[T, None, None]:
        for elt in progress_(self.progress_reporter, it, total):
            yield elt

    def _progress_start_(self, total: int):
        self.progress_reporter.start_(total)

    def _update_progress_(self, added_progress: int):
        self.progress_reporter.update_progress_(added_progress)

    def __call__(self, text: str, **kwargs) -> Dict[str, Any]:
        raise NotImplementedError()

    def supported_langs(self) -> Union[Set[str], Literal["any"]]:
        """
        :return: a list of supported languages, as ISO 639-3 codes, or
                 the string ``'any'``
        """
        return {"eng"}

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
    text: Optional[str]

    #: text split into blocks of texts. When dynamic blocks are given,
    #: the final network is dynamic, and split according to blocks.
    dynamic_blocks: Optional[List[Tuple[int, int]]] = None

    #: text splitted in tokens
    tokens: Optional[List[str]] = None
    #: mapping from a character to its corresponding token
    char2token: Optional[List[int]] = None
    #: text splitted into sentences, each sentence being a list of
    #: tokens
    sentences: Optional[List[List[str]]] = None

    #: quotes
    quotes: Optional[List[Quote]] = None
    #: quotes speakers
    speakers: Optional[List[Optional[Character]]] = None

    #: polarity of each sentence
    sentences_polarities: Optional[List[float]] = None

    #: NER entities
    entities: Optional[List[NEREntity]] = None

    #: coreference chains
    corefs: Optional[List[List[Mention]]] = None

    #: detected characters
    characters: Optional[List[Character]] = None

    #: character network (or list of network in the case of a dynamic
    #: network)
    character_network: Optional[Union[List[nx.Graph], nx.Graph]] = None

    # aliases of self.character_network
    def get_characters_graph(self) -> Optional[Union[List[nx.Graph], nx.Graph]]:
        return self.character_network

    characters_graph = property(get_characters_graph)
    character_graph = property(get_characters_graph)

    def get_character(
        self, name: str, partial_match: bool = True
    ) -> Optional[Character]:
        """Try to get a character by one of its name.

        .. note::

            Several characters may match the given name, but only the
            first one is returned.


        .. note::

            Comparison is case-insensitive.


        :param name: One of the name of the searched character.

        :param partial_match: when ``True``, will also return a
            character if the given ``name`` is only part of one of its
            name.  Otherwise, only a character with the given ``name``
            will be returned.

        :return: a :class:`.Character`, or ``None`` if no character
                 was found.
        """
        assert not self.characters is None
        # exact match
        for character in self.characters:
            if name.lower() in [n.lower() for n in character.names]:
                return character
        # partial match
        if partial_match:
            for character in self.characters:
                for cname in character.names:
                    if name.lower() in cname.lower():
                        return character
        # no match
        return None

    def export_graph_to_gexf(
        self,
        path: str,
        name_style: Union[
            Literal["longest", "shortest", "most_frequent"], Callable[[Character], str]
        ] = "most_frequent",
    ):
        """Export characters graph to Gephi's gexf format

        :param path: export file path
        :param name_style: see :func:`.graph_with_names`
            for more details
        """
        path = os.path.expanduser(path)
        if isinstance(self.character_network, list):
            G = dynamic_graph_to_gephi_graph(self.character_network)
            G = graph_with_names(G, name_style)
            # HACK: networkx cannot set a dynamic "weight" attribute
            # in gexf since "weight" has a specific meaning in
            # networkx. the following code hacks the XML tree
            # outputted by GEXFWriter to force the attribute name to
            # be "weight" (instead of "dweight", as outputted by
            # dynamic_graph_to_gephi_graph)
            writer = GEXFWriter()
            writer.add_graph(G)
            attribute_dweight = writer.xml.find(
                ".//graph/attributes/attribute[@title='dweight']"
            )
            dweight_old_id = attribute_dweight.get("id")
            attribute_dweight.set("id", "weight")
            attribute_dweight.set("title", "Weight")
            for attvalue in writer.xml.findall(
                f".//graph/edges/edge/attvalues/attvalue[@for='{dweight_old_id}']"
            ):
                attvalue.set("for", "weight")
            writer.write(path)
        else:
            G = graph_with_names(self.character_network, name_style)
            nx.write_gexf(G, path)

    def plot_graphs_to_dir(
        self,
        directory: str,
        name_style: Union[
            Literal["longest", "shortest", "most_frequent"], Callable[[Character], str]
        ] = "most_frequent",
        cumulative: bool = False,
        stable_layout: bool = False,
        layout: Optional[CharactersGraphLayout] = None,
        node_kwargs: Optional[List[Dict[str, Any]]] = None,
        edge_kwargs: Optional[List[Dict[str, Any]]] = None,
        label_kwargs: Optional[List[Dict[str, Any]]] = None,
        legend: bool = False,
    ):
        """Plot ``self.character_graph`` using reasonable default
        parameters, and save the produced figures in the specified
        directory.

        :param name_style: see :func:`.graph_with_names`
            for more details
        :param cumulative: if ``True`` plot a cumulative graph instead
            of a sequential one
        :param stable_layout: If this parameter is ``True``,
            characters will keep the same position in space at each
            timestep.  Characters' positions are based on the final
            cumulative graph layout.
        :param layout: pre-computed graph layout
        :param node_kwargs: passed to :func:`nx.draw_networkx_nodes`
        :param edge_kwargs: passed to :func:`nx.draw_networkx_nodes`
        :param label_kwargs: passed to :func:`nx.draw_networkx_labels`
        :param legend: passed to :func:`.plot_nx_graph_reasonably`
        """
        import matplotlib.pyplot as plt

        assert not self.character_network is None
        if isinstance(self.character_network, nx.Graph):
            raise ValueError("this function is supposed to be used on a dynamic graph")

        directory = directory.rstrip("/")
        directory = os.path.expanduser(directory)
        os.makedirs(directory, exist_ok=True)

        graphs = self.character_network
        if cumulative:
            graphs = cumulative_graph(self.character_network)

        if stable_layout:
            layout_graph = (
                graphs[-1]
                if cumulative
                else cumulative_graph(self.character_network)[-1]
            )
            layout = layout_nx_graph_reasonably(layout_graph)

        node_kwargs = node_kwargs or [{} for _ in range(len(self.character_network))]
        edge_kwargs = edge_kwargs or [{} for _ in range(len(self.character_network))]
        label_kwargs = label_kwargs or [{} for _ in range(len(self.character_network))]

        for i, G in enumerate(graphs):
            _, ax = plt.subplots()
            local_layout = layout
            if not local_layout is None:
                local_layout = layout_with_names(G, local_layout, name_style)
            G = graph_with_names(G, name_style=name_style)
            plot_nx_graph_reasonably(
                G,
                ax=ax,
                layout=local_layout,
                node_kwargs=node_kwargs[i],
                edge_kwargs=edge_kwargs[i],
                label_kwargs=label_kwargs[i],
                legend=legend,
            )
            plt.savefig(f"{directory}/{i}.png")
            plt.close()

    def plot_graph_to_file(
        self,
        path: str,
        name_style: Union[
            Literal["longest", "shortest", "most_frequent"], Callable[[Character], str]
        ] = "most_frequent",
        layout: Optional[CharactersGraphLayout] = None,
        fig: Optional[plt.Figure] = None,
        node_kwargs: Optional[Dict[str, Any]] = None,
        edge_kwargs: Optional[Dict[str, Any]] = None,
        label_kwargs: Optional[Dict[str, Any]] = None,
        tight_layout: bool = False,
        legend: bool = False,
    ):
        """Plot ``self.character_graph`` using reasonable parameters,
        and save the produced figure to a file

        :param name_style: see :func:`.graph_with_names`
            for more details
        :param layout: pre-computed graph layout
        :param fig: if specified, this matplotlib figure will be used
            for plotting
        :param node_kwargs: passed to :func:`nx.draw_networkx_nodes`
        :param edge_kwargs: passed to :func:`nx.draw_networkx_nodes`
        :param label_kwargs: passed to :func:`nx.draw_networkx_labels`
        :param tight_layout: if ``True``, will use matplotlib's tight_layout
        :param legend: passed to :func:`.plot_nx_graph_reasonably`
        """
        import matplotlib.pyplot as plt

        assert not self.character_network is None
        if isinstance(self.character_network, list):
            raise ValueError("this function is supposed to be used on a static graph")

        if not layout is None:
            layout = layout_with_names(self.character_network, layout, name_style)
        G = graph_with_names(self.character_network, name_style=name_style)
        if fig is None:
            # default values for a sufficiently sized graph
            fig = plt.gcf()
            assert not fig is None
            fig.set_dpi(300)
            fig.set_size_inches(24, 24)
        ax = fig.add_subplot(111)
        plot_nx_graph_reasonably(
            G,
            ax=ax,
            layout=layout,
            node_kwargs=node_kwargs,
            edge_kwargs=edge_kwargs,
            label_kwargs=label_kwargs,
            legend=legend,
        )
        if tight_layout:
            fig.tight_layout()
        plt.savefig(path)
        plt.close()

    def plot_graph(
        self,
        name_style: Union[
            Literal["longest", "shortest", "most_frequent"], Callable[[Character], str]
        ] = "most_frequent",
        fig: Optional[plt.Figure] = None,
        cumulative: bool = False,
        graph_start_idx: int = 1,
        stable_layout: bool = False,
        layout: Optional[CharactersGraphLayout] = None,
        node_kwargs: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        edge_kwargs: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        label_kwargs: Optional[Union[Dict[str, Any], List[Dict[str, Any]]]] = None,
        tight_layout: bool = False,
        legend: bool = False,
    ):
        """Plot ``self.character_network`` using reasonable default
        parameters

        .. note::

            when plotting a dynamic graph, a ``slider`` attribute is
            added to ``fig`` when it is given, in order to keep a
            reference to the slider.

        :param name_style: see :func:`.graph_with_names` for more
            details
        :param fig: if specified, this matplotlib figure will be used
            for plotting
        :param cumulative: if ``True`` and ``self.character_network``
            is dynamic, plot a cumulative graph instead of a
            sequential one
        :param graph_start_idx: When ``self.character_network`` is
            dynamic, index of the first graph to plot, starting at 1
            (not 0, since the graph slider starts at 1)
        :param stable_layout: if ``self.character_network`` is dynamic
            and this parameter is ``True``, characters will keep the
            same position in space at each timestep.  Characters'
            positions are based on the final cumulative graph layout.
        :param layout: pre-computed graph layout
        :param node_kwargs: passed to :func:`nx.draw_networkx_nodes`
        :param edge_kwargs: passed to :func:`nx.draw_networkx_nodes`
        :param label_kwargs: passed to :func:`nx.draw_networkx_labels`
        :param tight_layout: if ``True``, will use matplotlib's tight_layout
        :param legend: passed to :func:`.plot_nx_graph_reasonably`
        """
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider

        assert not self.character_network is None

        # self.character_network is a static graph
        if isinstance(self.character_network, nx.Graph):
            if not layout is None:
                layout = layout_with_names(self.character_network, layout, name_style)
            G = graph_with_names(self.character_network, name_style)
            if fig is None:
                # default value for a sufficiently sized graph
                fig = plt.gcf()
                assert not fig is None
                fig.set_dpi(300)
                fig.set_size_inches(24, 24)
            ax = fig.add_subplot(111)
            assert not isinstance(node_kwargs, list)
            assert not isinstance(edge_kwargs, list)
            assert not isinstance(label_kwargs, list)
            if tight_layout:
                fig.tight_layout()
            plot_nx_graph_reasonably(
                G,
                ax=ax,
                layout=layout,
                node_kwargs=node_kwargs,
                edge_kwargs=edge_kwargs,
                label_kwargs=label_kwargs,
                legend=legend,
            )
            return

        if not isinstance(self.character_network, list):
            raise TypeError
        # self.character_network is a list: plot a dynamic graph

        node_kwargs = node_kwargs or [{} for _ in range(len(self.character_network))]
        assert isinstance(node_kwargs, list)
        edge_kwargs = edge_kwargs or [{} for _ in range(len(self.character_network))]
        assert isinstance(edge_kwargs, list)
        label_kwargs = label_kwargs or [{} for _ in range(len(self.character_network))]
        assert isinstance(label_kwargs, list)

        if fig is None:
            fig, ax = plt.subplots()
            assert not fig is None
            fig.set_dpi(300)
            fig.set_size_inches(24, 24)
        else:
            ax = fig.add_subplot(111)
        assert not fig is None

        cumulative_character_networks = cumulative_graph(self.character_network)
        if stable_layout:
            layout = layout_nx_graph_reasonably(cumulative_character_networks[-1])

        def update(slider_value):
            assert isinstance(self.character_network, list)
            slider_i = int(slider_value) - 1

            character_networks = self.character_network
            if cumulative:
                character_networks = cumulative_character_networks

            G = character_networks[slider_i]

            local_layout = layout
            if not local_layout is None:
                local_layout = layout_with_names(G, local_layout, name_style)
            G = graph_with_names(G, name_style)

            ax.clear()
            plot_nx_graph_reasonably(
                G,
                ax=ax,
                layout=local_layout,
                node_kwargs=node_kwargs[slider_i],
                edge_kwargs=edge_kwargs[slider_i],
                label_kwargs=label_kwargs[slider_i],
                legend=legend,
            )
            ax.set_xlim(-1.2, 1.2)
            ax.set_ylim(-1.2, 1.2)

        slider_ax = fig.add_axes([0.1, 0.05, 0.8, 0.04])
        if tight_layout:
            fig.tight_layout()
        # HACK: we save the slider to the figure. This ensure the
        # slider is still alive at plotting time.
        fig.slider = Slider(  # type: ignore
            ax=slider_ax,
            label="Graph",
            valmin=1,
            valmax=len(self.character_network),
            valstep=[i + 1 for i in range(len(self.character_network))],
        )
        fig.slider.on_changed(update)  # type: ignore
        fig.slider.set_val(graph_start_idx)  # type: ignore


class Pipeline:
    """A flexible NLP pipeline"""

    #: all the possible parameters of the whole pipeline, that are
    #: shared between steps
    PipelineParameter = Literal["lang", "progress_reporter", "character_ner_tag"]

    def __init__(
        self,
        steps: List[PipelineStep],
        lang: str = "eng",
        progress_report: Optional[Literal["tqdm"]] = "tqdm",
        warn: bool = True,
    ) -> None:
        """
        :param steps: a ``tuple`` of :class:``PipelineStep``, that
            will be executed in order
        :param progress_report: if ``tqdm``, report the pipeline
            progress using tqdm.  if ``None``, does not report
            progress.
        :param lang: ISO 639-3 language code
        :param warn:
        """
        self.steps = steps

        self.progress_report: Optional[Literal["tqdm"]] = progress_report
        self.progress_reporter = get_progress_reporter(progress_report)

        self.lang = lang
        self.character_ner_tag = "PER"
        self.warn = warn

    def _pipeline_init_steps_(self, ignored_steps: Optional[List[str]] = None):
        """Initialise steps with global pipeline parameters.

        :param ignored_steps: a list of steps production.  All steps
            with a production in ``ignored_steps`` will be ignored.
        """
        steps_progress_reporter = self.progress_reporter.get_subreporter()
        steps = self._non_ignored_steps(ignored_steps)
        pipeline_params = {
            "progress_reporter": steps_progress_reporter,
            "character_ner_tag": self.character_ner_tag,
        }
        for step in steps:
            step_additional_params = step._pipeline_init_(self.lang, **pipeline_params)
            if not step_additional_params is None:
                for key, value in step_additional_params.items():
                    setattr(self, key, value)
                    pipeline_params[key] = value

    def _non_ignored_steps(
        self, ignored_steps: Optional[List[str]]
    ) -> List[PipelineStep]:
        """Get steps that are not ignored.

        :param ignored_steps: a list of steps production.  All steps
            with a production in ``ignored_steps`` wont be returned.
        """
        if ignored_steps is None:
            return self.steps
        return [
            s
            for s in self.steps
            if not any([p in s.production() for p in ignored_steps])
        ]

    def check_valid(
        self, *args, ignored_steps: Optional[List[str]] = None
    ) -> Tuple[bool, List[str]]:
        """Check that the current pipeline can be run, which is
        possible if all steps needs are satisfied

        :param args: list of additional attributes to add to the
            starting pipeline state.
        :param ignored_steps: a list of steps production.  All steps
            with a production in ``ignored_steps`` will be ignored.

        :return: a tuple : ``(True, [warnings])`` if the pipeline is
                 valid, ``(False, [errors])`` otherwise
        """

        pipeline_state = set(args).union({"text"})
        warnings = []

        steps = self._non_ignored_steps(ignored_steps)

        for i, step in enumerate(steps):
            if not step.needs().issubset(pipeline_state):
                return (
                    False,
                    [
                        "".join(
                            [
                                f"step {i + 1} ({step.__class__.__name__}) has unsatisfied needs. "
                                + f"needs: {step.needs()}. "
                                + f"available: {pipeline_state}). "
                                + f"missing: {step.needs() - pipeline_state}."
                            ]
                        ),
                    ],
                )

            if not step.optional_needs().issubset(pipeline_state):
                warnings.append(
                    "".join(
                        [
                            f"step {i + 1} ({step.__class__.__name__}) has unsatisfied optional needs. "
                            + f"needs: {step.optional_needs()}. "
                            + f"available: {pipeline_state}). "
                            + f"missing: {step.optional_needs() - pipeline_state}."
                        ]
                    )
                )

            pipeline_state = pipeline_state.union(step.production())

        return (True, warnings)

    def __call__(
        self,
        text: Optional[str] = None,
        ignored_steps: Optional[List[str]] = None,
        **kwargs,
    ) -> PipelineState:
        """Run the pipeline sequentially.

        :param ignored_steps: a list of steps production.  All steps
            with a production in ``ignored_steps`` will be ignored.

        :return: the output of the last step of the pipeline
        """
        is_valid, warnings_or_errors = self.check_valid(
            *kwargs.keys(), ignored_steps=ignored_steps
        )
        if not is_valid:
            raise ValueError(warnings_or_errors)
        if self.warn:
            for warning in warnings_or_errors:
                print(f"[warning] : {warning}", file=sys.stderr)

        self._pipeline_init_steps_(ignored_steps)

        state = PipelineState(text)
        # sets attributes to PipelineState dynamically. This ensures
        # that users can create steps returning custom attributes, and
        # that these attributes can be passed at pipeline call time.
        for key, value in kwargs.items():
            setattr(state, key, value)

        steps = self._non_ignored_steps(ignored_steps)

        for step in progress_(self.progress_reporter, steps):
            self.progress_reporter.update_message_(f"{step.__class__.__name__}")

            out = step(**state.__dict__)
            for key, value in out.items():
                setattr(state, key, value)

        return state

    def rerun_from(
        self,
        state: PipelineState,
        from_step: Union[str, Type[PipelineStep]],
        ignored_steps: Optional[List[str]] = None,
    ) -> PipelineState:
        """Recompute steps, starting from ``from_step`` (included).
        Previous steps results are not recomputed.

        .. note::

            steps are not re-inited using :func:`._pipeline_init_steps`.

        :param state: the previously computed state

        :param from_step: first step to recompute from.  Either :

                - ``str`` : in that case, the name of a step
                  production (``'tokens'``, ``'corefs'``...)

                - ``Type[PipelineStep]`` : in that case, the class of
                  a step

        :param ignored_steps: a list of steps production.  All steps
            with a production in ``ignored_steps`` will be ignored.

        :return: the output of the last step of the pipeline
        """
        steps = self._non_ignored_steps(ignored_steps)

        from_step_i = None
        for step_i, step in enumerate(steps):
            if step.__class__ == from_step or from_step in step.production():
                from_step_i = step_i
                break
        assert not from_step_i is None

        for step in progress_(self.progress_reporter, steps[from_step_i:]):
            self.progress_reporter.update_message_(f"{step.__class__.__name__}")
            out = step(**state.__dict__)
            for key, value in out.items():
                setattr(state, key, value)

        return state
