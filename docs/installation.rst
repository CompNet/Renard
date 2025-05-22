============
Installation
============

Using Pip
=========

Simply use ``pip install renard-pipeline``.

You can also install the following extras:

- ``stanza`` (``pip install renard-pipeline[stanza]``), for the Stanford CoreNLP pipeline
- ``spacy`` (``pip install renard-pipeline[spacy]``), for the Spacy
  coreference resolver Coreferee


Manual Installation
===================

The project uses `uv <https://docs.astral.sh/uv/>`_ to manage dependencies. Use :

- ``uv sync`` to install dependencies
- ``uv run python my_script.py`` to run a script under the virtual
  environment with dependencies


If you ever want to use the Stanford CoreNLP pipeline, you must install
``stanza`` (``pip install stanza``).

If you want to use the Spacy Coreferee coreference solver, you need to
install ``spacy``, ``spacy-transformers`` and ``corefereee`` (``pip
install spacy spacy-transformers coreferee``).
