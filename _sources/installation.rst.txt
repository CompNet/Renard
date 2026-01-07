============
Installation
============

Using Pip
=========

For the simplest case, use ``pip install renard-pipeline``. By default, this installs the CPU version of PyTorch. If you want GPU support to accelerate inference:

- CUDA 12.8: ``pip install renard-pipeline[cuda128]``
- ROCm 6.3: ``pip install renard-pipeline[rocm63]``


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
