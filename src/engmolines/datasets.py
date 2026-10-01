"""Bundled course data: measurements generated from stated models with documented noise and faults.

>>> from engmolines import datasets
>>> datasets.available()
['reactor_profiles.csv']
"""

from __future__ import annotations

from importlib import resources

INFO = {
    "reactor_profiles.csv": "Steady temperature profiles of a cooled tubular reactor at three inlet temperatures, with a dead thermocouple (project)."
}


def path(name: str):
    """Path-like object of a bundled data file (pass it to pandas.read_csv with comment='#')."""
    if name not in INFO:
        raise KeyError(f"Unknown dataset {name!r}. Available: {', '.join(sorted(INFO))}")
    return resources.files("engmolines.data") / name


def available() -> list[str]:
    """Names of the bundled data files."""
    return sorted(INFO)


def describe(name: str) -> str:
    """One-line description of a bundled data file."""
    return INFO[name]
