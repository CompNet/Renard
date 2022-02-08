============
Installation
============


Quick Start
===========

The project uses `Poetry <https://python-poetry.org/>`_ to manage dependencies. Use :

- ``poetry install`` to install dependencies
- ``poetry run python my_script.py`` to run a script under the virtual
  environment with dependencies
- ``poetry shell`` to get into a virtual environment with dependencies


If you ever want to use the Stanford CoreNLP pipeline, you can install
the ``stanza`` extra with ``poetry install -E stanza``.
