============
Installation
============

Using Pip
=========

For the simplest case, use ``pip install renard-pipeline``. By default, this installs the cuda version of PyTorch. For other versions:

CPU only: 
``pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install renard-pipeline``

ROCM 6.4:
``pip install torch --index-url https://download.pytorch.org/whl/rocm6.4 && pip install renard-pipeline``


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
