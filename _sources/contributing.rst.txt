============
Contributing
============

Contributions to Renard are welcome! You are encouraged to open an
issue if you encounter a problem or want to discuss a specific
feature. If you want to contribute a patch:

1. Check that your code matches our code quality guidelines and that
   all existing tests are passing with ``RENARD_TEST_ALL=1``.
2. Create a Github pull request with your patch, explaining the
   rationale behind it and giving a high level overview of your
   code. Mention the relevant issue if applicable.
3. We will discuss the contribution further and, hopefully, merge your
   contribution once the core maintainers are satisified.


Code Quality Guidelines
-----------------------

The source code of Renard is entirely typed. If possible, all
functions should be annotated with type information for arguments and
return types.

You should write docstrings for non-trivial functions and classes,
using the Spinx docstring format. Do not forget to add your
function/class/module to the ``docs/reference.rst`` file for it to
show up in the online documentation. If necessary, you can add new
documents to the documentation or complete existing ones.

Format all of your code using ``black``, so that the style stay
consistent in the repository. 

When relevant, it's better to write tests for your code. Tests live in
the ``tests`` directory. We use ``pytest`` to test code, and also use
``hypothesis`` when applicable. If you open a patch, make sure that
all tests are passing. In particular, do not rely on the CI, as it
does not run time costly tests! Check for yourself locally, using
``RENARD_TEST_ALL=1 python -m pytest tests``. Note that there are
specific tests and environment variable for optional dependencies such
as *stanza* (``RENARD_TEST_STANZA_OPTDEP``). These must be explicitely
set to ``1`` if you want to test optional dependencies, as
``RENARD_TEST_ALL=1`` does not enable test on these optional
dependencies.
