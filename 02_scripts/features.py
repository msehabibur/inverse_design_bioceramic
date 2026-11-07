"""
Shared composition featurizer, used by both train.py and screen.py so the
two scripts always produce identical feature columns.

Turns a chemical composition into numbers using three matminer featurizers:
  * ElementProperty (Magpie) - averages/spreads of elemental properties
  * Stoichiometry           - how the atomic fractions are distributed
  * ValenceOrbital          - fractions of s/p/d/f valence electrons
"""
from matminer.featurizers.base import MultipleFeaturizer
from matminer.featurizers.composition import (
    ElementProperty, Stoichiometry, ValenceOrbital,
)


def make_featurizer():
    f = MultipleFeaturizer([
        ElementProperty.from_preset("magpie", impute_nan=True),
        Stoichiometry(),
        ValenceOrbital(props=["frac"]),
    ])
    f.set_n_jobs(1)  # single-process: matminer multiprocessing hangs on macOS
    return f
