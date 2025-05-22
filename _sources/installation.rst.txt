============
Installation
============

Using Pip
=========

Simply use ``pip install renard-pipeline``.

Note that for some modules, you might need to install additional
libraries:

- ``stanza`` (``pip install stanza``), for the Stanford CoreNLP pipeline
- ``spacy`` (``pip install spacy coreferee``), for the Spacy
  coreference resolver Coreferee


Manual Installation
===================

The project uses `uv <https://docs.astral.sh/uv/>`_ to manage dependencies. Use :

- ``uv sync`` to install dependencies
- ``uv run python my_script.py`` to run a script under the virtual
  environment with dependencies
