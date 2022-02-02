# Renard

Relationships Extraction from NARrative Documents


# Dependencies

The project uses [Poetry](https://python-poetry.org/) to manage dependencies. Use :

- `poetry install` to install dependencies
- `poetry run python my_script.py` to run a script under the virtual environment with dependencies
- `poetry shell` to get into a virtual environment with dependencies


# Documentation

Documentation can be found at https://compnet.github.io/Renard/

If you need local documentation, it can be generated using `Sphinx`. From the `docs` directory, `make html` should create documentation under `docs/_build/html`. 

*Don't forget that you must be in the virtual environment of the project for the `make html` command to work : use `poetry run make html` if needed.*
