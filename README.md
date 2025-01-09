# Renard

[![DOI](https://joss.theoj.org/papers/10.21105/joss.06574/status.svg)](https://doi.org/10.21105/joss.06574)

Renard (Relationship Extraction from NARrative Documents) is a library for creating and using custom character networks extraction pipelines. Renard can extract dynamic as well as static character networks.

![The Renard logo](./docs/renard.svg)


# Installation

You can install the latest version using pip:

> pip install renard-pipeline

Currently, Renard supports Python>=3.8,<=3.11


# Documentation

Documentation, including installation instructions, can be found at https://compnet.github.io/Renard/

If you need local documentation, it can be generated using `Sphinx`. From the `docs` directory, `make html` should create documentation under `docs/_build/html`. 


# Tutorial

Renard's central concept is the `Pipeline`.A `Pipeline` is a list of `PipelineStep` that are run sequentially in order to extract a character graph from a document. Here is a simple example:

```python
from renard.pipeline import Pipeline
from renard.pipeline.tokenization import NLTKTokenizer
from renard.pipeline.ner import NLTKNamedEntityRecognizer
from renard.pipeline.character_unification import GraphRulesCharacterUnifier
from renard.pipeline.graph_extraction import CoOccurrencesGraphExtractor

with open("./my_doc.txt") as f:
	text = f.read()

pipeline = Pipeline(
	[
		NLTKTokenizer(),
		NLTKNamedEntityRecognizer(),
		GraphRulesCharacterUnifier(min_appearance=10),
		CoOccurrencesGraphExtractor(co_occurrences_dist=25)
	]
)

out = pipeline(text)
```

For more information, see `renard_tutorial.py`, which is a tutorial in the `jupytext` format. You can open it as a notebook in Jupyter Notebook (or export it as a notebook with `jupytext --to ipynb renard-tutorial.py`).



# Running tests 

`Renard` uses `pytest` for testing. To launch tests, use the following command : 

> poetry run python -m pytest tests

Expensive tests are disabled by default. These can be run by setting the environment variable `RENARD_TEST_ALL` to `1`.


# Contributing

see [the "Contributing" section of the documentation](https://compnet.github.io/Renard/contributing.html).


# How to cite

If you use Renard in your research project, please cite it as follows:

```bibtex
@Article{Amalvy2024,
  doi	       = {10.21105/joss.06574},
  year	       = {2024},
  publisher    = {The Open Journal},
  volume       = {9},
  number       = {98},
  pages	       = {6574},
  author       = {Amalvy, A. and Labatut, V. and Dufour, R.},
  title	       = {Renard: A Modular Pipeline for Extracting Character
                  Networks from Narrative Texts},
  journal      = {Journal of Open Source Software},
} 
```

We would be happy to hear about your usage of Renard, so don't hesitate to reach out!
