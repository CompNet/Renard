from typing import Any, Type, Union, Optional
import os
from dataclasses import dataclass
from renard.pipeline import Pipeline, PipelineStep
from renard.pipeline.core import PipelineState
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
from renard.graph_utils import graph_with_names
from renard.resources.novels import load_novel
import matplotlib.pyplot as plt
import matplotlib.colors
import networkx as nx
from pyvis.network import Network
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
    PipelineStepBuilder(
        NLTKNamedEntityRecognizer, "Fast NER method relying on POS tagging", {}
    ),
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
    last_run_state: Optional[PipelineState] = None


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


def get_viridis_hex(value: float, min_value: float, max_value: float) -> str:
    value = max(min(value, max_value), min_value)
    norm_value = (value - min_value) / (max_value - min_value)
    color = plt.cm.viridis(norm_value)  # type: ignore
    return matplotlib.colors.to_hex(color)


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
) -> tuple[str, list[list], list[list], str]:
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

    # plotting, see https://github.com/gradio-app/gradio/issues/4574
    G = graph_with_names(out.character_network, name_style="most_frequent")
    max_degree = max(v for _, v in G.degree) if len(G.degree) > 0 else 0
    for u in G.nodes:
        G.nodes[u]["size"] = 10 + G.degree[u]
        G.nodes[u]["color"] = get_viridis_hex(G.degree[u], 0, max_degree)
        maybe_char = out.get_character(u)
        if maybe_char is None:
            continue
        assert not maybe_char is None
        G.nodes[u]["title"] = "aliases: " + ", ".join(maybe_char.names)

    net = Network(width="100%")
    net.from_nx(G)
    # NOTE: layout found to have good default with "Pride and
    # Prejudice" with default parameters
    net.options.physics.use_barnes_hut(
        {
            "gravity": -22000,
            "central_gravity": 4.9,
            "spring_length": 95,
            "spring_strength": 0.04,
            "damping": 0.09,
            "overlap": 0,
        }
    )
    for u in net.nodes:
        u["font"] = {"size": max(12, 2 * G.degree[u["id"]])}
    net.show_buttons(filter_=["physics"])
    net.set_template(f"{os.path.dirname(__file__)}/pyvis_template.html")
    html = net.generate_html()
    html = html.replace("'", '"')
    html = f"""<iframe style="width: 100%; height: 650px;" name="Character Network" srcdoc='{html}'></iframe>"""

    out_path = f"{request.session_hash}.graphml"
    nx.write_graphml(G, out_path)

    return (
        html,
        # name    # occurrences    # aliases
        [[mfn(c), len(c.mentions), ", ".join(c.names)] for c in out.characters],
        # char 1,  char 2
        [[mfn(c1), mfn(c2)] for c1, c2 in out.character_network.edges],
        out_path,
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
    with gr.Column():
        # Inputs
        with gr.Row():
            with gr.Column():
                with gr.Group():
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
                            gr.Markdown(step_builder.description)
                            render_kwargs_(step_builder, tok_kwargs)

                with gr.Group():
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
                            ner_builder = select_step_builder(
                                ner_step_builders, ner_step
                            )
                            gr.Markdown(ner_builder.description)
                            render_kwargs_(ner_builder, ner_kwargs)

                with gr.Group():
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
                            gr.Markdown(step_builder.description)
                            render_kwargs_(step_builder, cu_kwargs)

                with gr.Group():
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
                            gr.Markdown(step_builder.description)
                            render_kwargs_(step_builder, ge_kwargs)

            with gr.Column(scale=3):
                # TODO: pipeline level parameter like 'lang'
                input_text = gr.State()
                input_text_radio = gr.Radio(
                    choices=["Predefined Example", "Raw text", "Upload .txt file"],
                    label="Input type",
                    value="Predefined Example",
                )

                # NOTE: for some reason, tabs have an issue where the
                # component in the second tab is invisible.
                @gr.render(inputs=input_text_radio)
                def render_input_text(input_type: str):
                    if input_type == "Predefined Example":
                        pp = load_novel("pride_and_prejudice")
                        text_area = gr.TextArea(
                            label="Pride and Prejudice", value=pp, interactive=False
                        )
                        input_text.value = pp
                    elif input_type == "Upload .txt file":
                        upload_area = gr.File(
                            label="Input .txt file", file_types=["text"]
                        )
                        upload_area.upload(
                            lambda path: open(path).read(), [upload_area], [input_text]
                        )
                    elif input_type == "Raw text":
                        text_area = gr.TextArea(label="Input text")
                        text_area.change(lambda text: text, [text_area], [input_text])
                    else:
                        print(f"Unknown input type: {input_type}")

                run_btn = gr.Button()

        # Outputs
        with gr.Column():
            with gr.Group():
                gr.Markdown("## Character Network")
                out_plot = gr.HTML(label="Character Network")
                out_dl_btn = gr.DownloadButton("Download as .graphml")

            with gr.Tab(label="Characters"):
                out_character_df = gr.Dataframe(
                    headers=["name", "occurrences", "aliases"],
                    datatype=["str", "number", "str"],
                    type="array",
                    label="Characters",
                )
            with gr.Tab(label="Edges"):
                out_edge_df = gr.DataFrame(
                    headers=["character 1", "character 2"],
                    datatype=["str", "str"],
                    type="array",
                    label="Edges",
                )

        run_btn.click(
            fn=run_pipeline,
            inputs=[
                input_text,
                tok_ddown,
                tok_kwargs,
                ner_ddown,
                ner_kwargs,
                cu_ddown,
                cu_kwargs,
                ge_ddown,
                ge_kwargs,
            ],
            outputs=[out_plot, out_character_df, out_edge_df, out_dl_btn],
        )

    demo.load(init_pipeline_)
    demo.unload(free_pipeline_)

demo.launch()
