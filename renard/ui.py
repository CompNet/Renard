from typing import Any, Type
from dataclasses import dataclass

from tibert import Union
from renard.pipeline import Pipeline, PipelineStep
from renard.pipeline.ner import BertNamedEntityRecognizer
from renard.pipeline.tokenization import NLTKTokenizer
from renard.pipeline.ner import NLTKNamedEntityRecognizer
from renard.pipeline.corefs import BertCoreferenceResolver
from renard.pipeline.character_unification import (
    Character,
    GraphRulesCharacterUnifier,
    NaiveCharacterUnifier,
)
from renard.pipeline.graph_extraction import CoOccurrencesGraphExtractor
import networkx as nx
import matplotlib.pyplot as plt
import gradio as gr


@dataclass
class PipelineStepBuilder:
    cls: Type[PipelineStep]
    description: str
    # { name => (type, default value) }
    kwargs: dict[str, tuple[Union[Type, str], Any]]

    def name(self) -> str:
        return self.cls.__name__

    def make_step(self, step_kwargs: dict[str, Any]) -> PipelineStep:
        return self.cls(**step_kwargs)


tokenization_step_builders = [PipelineStepBuilder(NLTKTokenizer, "", {})]
ner_step_builders = [
    PipelineStepBuilder(NLTKNamedEntityRecognizer, "Fast statistical NER model", {}),
    PipelineStepBuilder(
        BertNamedEntityRecognizer, "BERT based deep learning NER model", {}
    ),
]
coref_step_builders = [
    PipelineStepBuilder(
        BertCoreferenceResolver, "BERT based deep learning coreference model", {}
    )
]
character_unification_step_builders = [
    PipelineStepBuilder(
        GraphRulesCharacterUnifier,
        "Rules based character unification algorithm",
        {
            "min_appearances": ("uint", 0),
            "link_corefs_mentions": (bool, False),
            "ignore_leading_determiner": (bool, False),
        },
    ),
    PipelineStepBuilder(NaiveCharacterUnifier, "Baseline naive character unifier", {}),
]
graph_extraction_step_builder = [
    PipelineStepBuilder(
        CoOccurrencesGraphExtractor,
        "",
        {"co_occurrences_dist": ("uint", 25)},
    )
]


def select_step_builder(
    builders: list[PipelineStepBuilder], name: str
) -> PipelineStepBuilder:
    return next(b for b in builders if b.name() == name)


@dataclass
class UserState:
    pipeline: Pipeline
    last_run_kwargs: list[dict]


user_state: dict[str, UserState] = {}


def init_pipeline_(request: gr.Request):
    if not request.session_hash:
        return
    global user_state
    default_pipeline = Pipeline(
        [
            NLTKTokenizer(),
            NLTKNamedEntityRecognizer(),
            GraphRulesCharacterUnifier(),
            CoOccurrencesGraphExtractor(co_occurrences_dist=25),
        ]
    )
    user_state[request.session_hash] = UserState(default_pipeline, [])


def free_pipeline_(request: gr.Request):
    global user_pipelines
    if request.session_hash in user_state:
        del user_state[request.session_hash]


def mfn(character: Character) -> str:
    name = character.most_frequent_name()
    if name is None:
        return "?"
    return name


def update_pipeline(
    state: UserState,
    step_builders: list[PipelineStepBuilder],
    step_kwargs: list[dict[str, Any]],
):
    step_names = [b.name() for b in step_builders]
    pipeline_names = [step.__class__.__name__ for step in state.pipeline.steps]
    if step_names == pipeline_names and step_kwargs == state.last_run_kwargs:
        return state.pipeline

    pipeline = Pipeline(
        [
            builder.make_step(kwargs)
            for builder, kwargs in zip(step_builders, step_kwargs)
        ]
    )
    return pipeline


def run_pipeline(
    request: gr.Request,
    text: str,
    tokenization_step: str,
    tokenization_kwargs: dict[str, Any],
    ner_step: str,
    ner_kwargs: dict[str, Any],
    character_unification_step: str,
    character_unification_kwargs: dict[str, Any],
    graph_extraction_step: str,
    graph_extraction_kwargs: dict[str, Any],
) -> tuple[plt.Figure, list[list], list[list]]:
    global user_state
    assert request.session_hash
    pipeline = user_state[request.session_hash]
    kwargs = [
        tokenization_kwargs,
        ner_kwargs,
        character_unification_kwargs,
        graph_extraction_kwargs,
    ]
    pipeline = update_pipeline(
        pipeline,
        [
            select_step_builder(tokenization_step_builders, tokenization_step),
            select_step_builder(ner_step_builders, ner_step),
            select_step_builder(
                character_unification_step_builders, character_unification_step
            ),
            select_step_builder(graph_extraction_step_builder, graph_extraction_step),
        ],
        kwargs,
    )
    user_state[request.session_hash].pipeline = pipeline
    user_state[request.session_hash].last_run_kwargs = kwargs

    out = pipeline(text)
    assert not out.characters is None
    assert not out.character_network is None
    assert isinstance(out.character_network, nx.Graph)

    fig, ax = plt.subplots()
    out.plot_graph(name_style="most_frequent", fig=fig)

    return (
        fig,
        # name    # occurrences    # aliases
        [[mfn(c), len(c.mentions), ", ".join(c.names)] for c in out.characters],
        # char 1,  char 2
        [[mfn(c1), mfn(c2)] for c1, c2 in out.character_network.edges],
    )


def render_kwargs_(step_builder: PipelineStepBuilder, kwargs: gr.State):
    def set_kwargs(kwargs: dict, name: str, value: Any):
        return {**kwargs, name: value}

    for name, (typ, default) in step_builder.kwargs.items():
        key = f"{step_builder.name()}-{name}"
        if typ == str:
            tbox = gr.Textbox(label=name, value=default, interactive=True, key=key)
            tbox.change(set_kwargs, [kwargs, gr.State(name), tbox], [kwargs])
        elif typ == int:
            nb = gr.Number(label=name, value=float(default), interactive=True, key=key)
            nb.change(set_kwargs, [kwargs, gr.State(name), nb], [kwargs])
        elif typ == "uint":
            uint = gr.Number(
                label=name, value=float(default), minimum=0, interactive=True, key=key
            )
            uint.change(set_kwargs, [kwargs, gr.State(name), uint], [kwargs])
        elif typ == bool:
            chk = gr.Checkbox(label=name, value=default, key=key)
            chk.change(set_kwargs, [kwargs, gr.State(name), chk], [kwargs])
        else:
            print(f"unknown kwarg type for {name}: {typ}")


with gr.Blocks(title="Renard") as demo:
    with gr.Row():
        # Inputs
        with gr.Column():
            with gr.Group("Tokenization"):
                gr.Markdown("## Step 1: Tokenization")
                with gr.Accordion("Click to expand", open=False):
                    tok_kwargs = gr.State({})
                    tok_ddown = gr.Dropdown(
                        [s.name() for s in tokenization_step_builders],
                        value=tokenization_step_builders[0].name(),
                        label="Tokenization step",
                    )
                    tok_ddown.change(lambda: {}, [], [tok_kwargs])

                    @gr.render(inputs=tok_ddown)
                    def render_tok_kwargs(tok_step: str):
                        step_builder = select_step_builder(
                            tokenization_step_builders, tok_step
                        )
                        render_kwargs_(step_builder, tok_kwargs)

            with gr.Group("NER"):
                gr.Markdown("## Step 2: Named Entity Recognition")
                with gr.Accordion("Click to expand", open=False):
                    ner_kwargs = gr.State({})
                    ner_ddown = gr.Dropdown(
                        [s.name() for s in ner_step_builders],
                        value=ner_step_builders[0].name,
                        label="NER step",
                    )
                    ner_ddown.change(lambda: {}, [], [ner_kwargs])

                    @gr.render(inputs=ner_ddown)
                    def render_ner_kwargs(ner_step: str):
                        ner_builder = select_step_builder(ner_step_builders, ner_step)
                        render_kwargs_(ner_builder, ner_kwargs)

            with gr.Group("Character Unification"):
                gr.Markdown("## Step 3: Character Unification")
                with gr.Accordion("Click to expand", open=False):
                    cu_kwargs = gr.State({})
                    cu_ddown = gr.Dropdown(
                        [s.name() for s in character_unification_step_builders],
                        value=character_unification_step_builders[0].name(),
                        label="Character unification step",
                    )
                    cu_ddown.change(lambda: {}, [], [cu_kwargs])

                    @gr.render(inputs=cu_ddown)
                    def render_cu_kwargs(cu_step: str):
                        step_builder = select_step_builder(
                            character_unification_step_builders, cu_step
                        )
                        render_kwargs_(step_builder, cu_kwargs)

            with gr.Group("Graph Extraction"):
                gr.Markdown("## Step 4: Graph Extraction")
                with gr.Accordion("Click to expand", open=False):
                    ge_kwargs = gr.State({})
                    ge_ddown = gr.Dropdown(
                        [s.name() for s in graph_extraction_step_builder],
                        value=graph_extraction_step_builder[0].name(),
                        label="Graph extraction step",
                    )
                    ge_ddown.change(lambda: {}, [], [ge_kwargs])

                    @gr.render(inputs=ge_ddown)
                    def render_ge_kwargs(ge_step: str):
                        step_builder = select_step_builder(
                            graph_extraction_step_builder, ge_step
                        )
                        render_kwargs_(step_builder, ge_kwargs)

            # TODO: pipeline level parameter like 'lang'
            text = gr.TextArea()
            run_btn = gr.Button()

        # Outputs
        with gr.Column():
            out_plot = gr.Plot()
            out_character_df = gr.Dataframe(
                headers=["name", "occurrences", "aliases"],
                datatype=["str", "number", "str"],
                type="array",
                label="characters",
            )
            out_edge_df = gr.DataFrame(
                headers=["character 1", "character 2"],
                datatype=["str", "str"],
                type="array",
                label="edges",
            )
        run_btn.click(
            fn=run_pipeline,
            inputs=[
                text,
                tok_ddown,
                tok_kwargs,
                ner_ddown,
                ner_kwargs,
                cu_ddown,
                cu_kwargs,
                ge_ddown,
                ge_kwargs,
            ],
            outputs=[out_plot, out_character_df, out_edge_df],
        )

    demo.load(init_pipeline_)
    demo.unload(free_pipeline_)

demo.launch()
