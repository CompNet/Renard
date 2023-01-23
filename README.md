# Renard

Relationships Extraction from NARrative Documents


# Documentation

Documentation, including installation instructions, can be found at https://compnet.github.io/Renard/

If you need local documentation, it can be generated using `Sphinx`. From the `docs` directory, `make html` should create documentation under `docs/_build/html`. 


# Running tests 

`Renard` uses `pytest` for testing. To launch tests, use the following command : 

> poetry run python -m pytest tests

Expensive tests are disabled by default. These can be run by setting the environment variable `RENARD_TEST_ALL` to `1`.
