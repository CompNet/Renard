from typing import Any, Type
from dataclasses import dataclass
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
    kwargs: dict[str, Type]

    def name(self) -> str:
        return self.cls.__name__

    def make_step(self, step_kwargs: dict[str, Any]) -> PipelineStep:
        return self.cls(**step_kwargs)


tokenization_step_builders = [PipelineStepBuilder(NLTKTokenizer, {})]
ner_step_builders = [
    PipelineStepBuilder(NLTKNamedEntityRecognizer, {}),
    PipelineStepBuilder(BertNamedEntityRecognizer, {}),
]
coref_step_builders = [PipelineStepBuilder(BertCoreferenceResolver, {})]
character_unification_step_builders = [
    PipelineStepBuilder(GraphRulesCharacterUnifier, {}),
    PipelineStepBuilder(NaiveCharacterUnifier, {}),
]
graph_extraction_step_builder = [PipelineStepBuilder(CoOccurrencesGraphExtractor, {})]


def select_step_builder(
    builders: list[PipelineStepBuilder], name: str
) -> PipelineStepBuilder:
    return next(b for b in builders if b.name() == name)


user_pipelines = {}


def init_pipeline_(request: gr.Request):
    global user_pipelines
    default_pipeline = Pipeline(
        [
            NLTKTokenizer(),
            NLTKNamedEntityRecognizer(),
            GraphRulesCharacterUnifier(),
            CoOccurrencesGraphExtractor(co_occurrences_dist=25),
        ]
    )
    user_pipelines[request.session_hash] = default_pipeline


def free_pipeline_(request: gr.Request):
    global user_pipelines
    if request.session_hash in user_pipelines:
        del user_pipelines[request.session_hash]


def mfn(character: Character) -> str:
    name = character.most_frequent_name()
    if name is None:
        return "?"
    return name


def update_pipeline(
    pipeline: Pipeline,
    step_builders: list[PipelineStepBuilder],
    step_kwargs: list[dict[str, Any]],
):
    # TODO: does not take into account kwargs changes
    step_names = [b.cls.__name__ for b in step_builders]
    if step_names == [step.__class__.__name__ for step in pipeline.steps]:
        return pipeline
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
    ner_step: str,
    character_unification_step: str,
) -> tuple[plt.Figure, list[list], list[list]]:
    pipeline = user_pipelines[request.session_hash]
    pipeline = update_pipeline(
        pipeline,
        [
            select_step_builder(tokenization_step_builders, tokenization_step),
            select_step_builder(ner_step_builders, ner_step),
            select_step_builder(
                character_unification_step_builders, character_unification_step
            ),
            # TODO: deal with graph extractor kwargs
            select_step_builder(
                graph_extraction_step_builder, "CoOccurrencesGraphExtractor"
            ),
        ],
        # TODO:
        [{} for _ in range(3)],
    )
    user_pipelines[request.session_hash] = pipeline
    print(pipeline.steps)

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


with gr.Blocks(title="Renard") as demo:
    with gr.Row():
        # Inputs
        with gr.Column():
            tok_radio = gr.Radio(
                [s.name() for s in tokenization_step_builders],
                value="NLTKTokenizer",
                label="Tokenization step",
            )
            ner_radio = gr.Radio(
                [s.name() for s in ner_step_builders],
                value="NLTKNamedEntityRecognizer",
                label="NER step",
            )
            cu_radio = gr.Radio(
                [s.name() for s in character_unification_step_builders],
                value="GraphRulesCharacterUnifier",
                label="Character unification step",
            )
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
            inputs=[text, tok_radio, ner_radio, cu_radio],
            outputs=[out_plot, out_character_df, out_edge_df],
        )

    demo.load(init_pipeline_)
    demo.unload(free_pipeline_)

demo.launch()
