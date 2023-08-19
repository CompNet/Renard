# Renard

Relationships Extraction from NARrative Documents


# Installation

You can install the latest version using pip:

> pip install renard-pipeline


# Documentation

Documentation, including installation instructions, can be found at https://compnet.github.io/Renard/

If you need local documentation, it can be generated using `Sphinx`. From the `docs` directory, `make html` should create documentation under `docs/_build/html`. 


# Tutorial

`renard_tutorial.py` is a tutorial in the `jupytext` format. You can open it as a notebook in Jupyter Notebook (or export it as a notebook with `jupytext --to ipynb renard-tutorial.py`).


# Running tests 

`Renard` uses `pytest` for testing. To launch tests, use the following command : 

> poetry run python -m pytest tests

Expensive tests are disabled by default. These can be run by setting the environment variable `RENARD_TEST_ALL` to `1`.
