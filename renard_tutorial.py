# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Renard : Relationships Extraction from NARrative Documents
#
#
# Renard is a modular python pipeline that can extract static or dynamic character networks from narrative documents.
#
# - Installation: `pip install renard-pipeline`
# - Documentation: https://compnet.github.io/Renard/

# %% [markdown]
# # General Overview
#
#
# The central object in Renard is a `Pipeline` that you can execute on a document. A `Pipeline` is formed by a sequential series of natural language processing `Step`s needed to extract a character network. Here is an example of a classic `Pipeline`:
#
# ```
#                text
#                 |
#                 v
#           [tokenization]
#                 |
#                 v
#    [named entity recognition (NER)]
#                 |
#                 v
#       [characters unification]
#                 |
#                 v
#    [co-occurences graph extraction]
#                 |
#                 v
#          character network
# ```
#
# Which you could write using Renard:
#
# ```python
# from renard.pipeline import Pipeline
# from renard.pipeline.tokenization import NLTKTokenizer
# from renard.pipeline.ner import NLTKNamedEntityRecognizer
# from renard.pipeline.character_unification import GraphRulesCharacterUnifier
# from renard.pipeline.graph_extraction import CoOccurrencesGraphExtractor
#
# # Pipeline definition
# pipeline = Pipeline(
#     [
#         NLTKTokenizer(),                                                 # tokenization
#         NLTKNamedEntityRecognizer(),                                     # named entity recognition
#         GraphRulesCharactersExtractor(),                                 # characters extraction
#         CoOccurrencesGraphExtractor(co_occurrences_dist=(1, "sentences")) # graph extraction
#     ]
# )
# ```
#
# You can then execute that pipeline on a given text:
#
# ```python
# with open("./my_text.txt") as f:
#     text = f.read()
#
# out = pipeline(text)
# ```
#
# The `out` object then contains the pipeline execution's result, which includes the character network (see the `characters_graph` attribute). We can for example export this network on the disk:
#
# ```python
# out.export_graph_to_gexf("./my_graph.gexf")
# ```

# %% [markdown]
# # Example: Extracting a Character Network with Existing NER Annotations

# %% [markdown]
# ## Static Graph Extraction

# %%
from renard.pipeline import Pipeline
from renard.pipeline.character_unification import GraphRulesCharacterUnifier
from renard.pipeline.graph_extraction import CoOccurrencesGraphExtractor
from renard.utils import load_conll2002_bio

# the utility function "load_conll2002_bio" allows loading BIO NER
# annotations from disk.
sentences, tokens, entities = load_conll2002_bio("./tests/three_musketeers_fra.bio")

# pipeline creation. Only the characters extraction and graph
# extraction steps are specified, since tokenization and BIO tags are
# already given.
pipeline = Pipeline(
    [
        GraphRulesCharacterUnifier(),
        # an interaction will be a co-occurence in a range of 3
        # sentences or less
        CoOccurrencesGraphExtractor(co_occurrences_dist=(3, "sentences")),
    ],
    lang="fra",
)

# pipeline execution. The caller gives tokenization and NER entities
# to the pipeline at runtime
out = pipeline(tokens=tokens, sentences=sentences, entities=entities)

# %% [markdown]
# ## Graph Display
#
# Renard can display the extracted graph using `matplotlib`. This visualization is naive, and it's recommended to export the graph and then use a software such as `Gephi` for more advanced usage.
#
# _note : if there are display issues with Jupyter see https://discourse.jupyter.org/t/jupyter-notebook-zmq-message-arrived-on-closed-channel-error/17869/27. `pip install --upgrade "pyzmq<25" "jupyter_client<8"` might fix the issue._
#
# _note : if there are display issues, you can still save an image on disk with `out.plot_graph_to_file("./my_image.png")`_

# %%
# %matplotlib notebook
import matplotlib.pyplot as plt

out.plot_graph()
plt.show()

# %% [markdown]
# # Extraction Setup
#
# Here are a few examples of tweaks you can apply in the extraction setup. Depending on your text, these can enhance the quality of the extracted graph.

# %%
pipeline = Pipeline(
    [
        # at least 3 occurences of a characters are needed for them to
        # be included in the graph (default is 1)
        GraphRulesCharacterUnifier(min_appearances=3),
        # A co-occurence between two characters is counted if its
        # range is lower or equal to 10 sentences
        CoOccurrencesGraphExtractor(co_occurrences_dist=(10, "sentences")),
    ],
    lang="fra",
)

out = pipeline(tokens=tokens, sentences=sentences, entities=entities)
out.plot_graph()
plt.show()

# %% [markdown]
# ## Advanced Graph Manipulation
#
# The `characters_graph` attribute contains the `networkx` graph extracted by `Renard`. It is possible to manipulate this graph directly using python for advanced usage.

# %%
import networkx as nx

print(nx.density(out.characters_graph))

# %% [markdown]
# ## Graphi Export
#
# The `export_graph_to_gexf` function can export the graph to the Gephi format.

# %%
out.export_graph_to_gexf("./my_graph.gexf")

# %% [markdown]
# ## Extraction d'un graphe dynamique
#
# It is possible to ask the `CoOccurrencesGraphExtractor` step to extract a _dynamic_ graph using the `dynamic` argument and a few parameters.

# %%
pipeline = Pipeline(
    [
        GraphRulesCharacterUnifier(min_appearances=3),
        CoOccurrencesGraphExtractor(
            co_occurrences_dist=(20, "sentences"),
            dynamic=True,  # we want to extract a dynamic graph (i.e. a list of sequential graphs)
            dynamic_window=20,  # the size, in interaction, of each graph
            dynamic_overlap=0,  # overlap between windows
        ),
    ],
    lang="fra",
)

out = pipeline(tokens=tokens, sentences=sentences, entities=entities)

# display adapts to the fact that the extracted graph is dynamic, and
# allow exploration of each graph of the list.
out.plot_graph()
plt.show()

# %% [markdown]
# It is also possible to explore the cumulative dynamic graph:

# %%
out.plot_graph(cumulative=True, stable_layout=True)
plt.show()

# %% [markdown]
# And to export the dynamic graph to the Gephi format. When doing so,
# the 'timeline' feature of Gephi will work as expected:

# %%
out.export_graph_to_gexf("./graphe_dynamique.gexf")
