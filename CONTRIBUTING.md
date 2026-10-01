# Contributing

Contributions to engmolines are welcome - bug reports, new validation cases, notebooks, exercises and methods.

- **Bugs:** please open an issue with a minimal example (data, code, expected and actual result).
- **New methods:** include a docstring stating the algorithm and its reference, unit tests, and at least one
  validation check in `docs/validate.py` against an independent reference (an exact solution, a published
  worked example, or an independent implementation).
- **Notebooks** are generated from `notebooks/build_notebooks.py`; edit that file and run
  `python notebooks/build_notebooks.py`, never the `.ipynb` files directly.
- Run `pytest`, `python docs/validate.py` and `ruff check .` before submitting.
