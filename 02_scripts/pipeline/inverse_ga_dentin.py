"""
Showcase 2 - Step 3.  Genetic-algorithm inverse design targeting DENTIN.

Targets:  Young's modulus 20 GPa, Vickers hardness 0.7 GPa, density 2.1 g/cm^3.

Same GA architecture as inverse_ga.py: pymoo single-objective, charge-balanced
oxide decoding.  Two things change vs the enamel showcase:

  (1) Cation pool restricted to bone-mineral elements
      {Ca, P, Si, Mg, Na, Sr, K, Al}.  No Zr/Ti/Ce/Y/La/Ce because they sit far
      from the chemistry of physiological mineral.

  (2) Property models are the DENTIN-specific ones trained on the N=531
      biocompatible-oxide slice (model_*_dentin.joblib).

NOTE: thermodynamic stability is NOT enforced -- and the dentin V/atom helper is
only moderate (CV R^2 ~ 0.60), so density predictions carry larger uncertainty
than the enamel showcase.  Designed candidates need a DFT/MLFF follow-up.

Output: designed_dentin_compositions.csv
"""
import os
from math import gcd
from functools import reduce
import numpy as np
import pandas as pd
import joblib
from pymatgen.core import Composition
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # noqa: make paths/features importable
from features import make_featurizer
from paths import DATA, MODELS
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize

# --- dentin mechanical fingerprint ---
TARGET_E, TARGET_H, TARGET_RHO = 20.0, 0.7, 2.1
W_E, W_H, W_RHO = 0.588, 0.0, 0.412   # hardness EXCLUDED from objective (weak R2, collinear with E)      # same weights as enamel for comparability

# --- bone-mineral cation pool + typical oxidation states ---
CHARGE = {"Ca": 2, "P": 5, "Si": 4, "Mg": 2,
          "Na": 1, "Sr": 2, "K": 1, "Al": 3}
CATIONS = list(CHARGE)

MAX_COUNT = 6      # largest count of any single cation in a formula unit
SEEDS = [42, 1, 7, 13, 21]   # repeat the search; keep best per distinct formula

mod  = joblib.load(os.path.join(MODELS, "model_modulus_dentin.joblib"))
hard = joblib.load(os.path.join(MODELS, "model_hardness_dentin.joblib"))
vol  = joblib.load(os.path.join(MODELS, "model_volperatom_dentin.joblib"))
featurizer = make_featurizer()
AMU_PER_A3_TO_GCC = 1.66054


def decode(x):
    """GA genome -> small-integer, charge-balanced oxide (see inverse_ga.py)."""
    amt = dict(zip(CATIONS, x))
    top = sorted(amt.items(), key=lambda kv: -kv[1])[:4]
    keep = {e: a for e, a in top if a > 0.10}
    if not keep:
        keep = {"Ca": 1.0}
    lo = min(keep.values())
    counts = {e: max(1, min(MAX_COUNT, int(round(a / lo)))) for e, a in keep.items()}
    o2 = sum(n * CHARGE[e] for e, n in counts.items())      # 2 * oxygen
    if o2 % 2:
        counts = {e: 2 * n for e, n in counts.items()}
        o2 *= 2
    atoms = {**counts, "O": o2 // 2}
    g = reduce(gcd, atoms.values())
    atoms = {e: n // g for e, n in atoms.items()}
    return Composition(atoms)


def score_batch(comps):
    d = pd.DataFrame({"composition_obj": comps})
    d = featurizer.featurize_dataframe(d, col_id="composition_obj",
                                       ignore_errors=True)
    cc = vol["features"]
    d[cc] = d[cc].fillna(d[cc].mean())
    vpa = vol["model"].predict(d[cc])
    mm  = np.array([c.weight / c.num_atoms for c in comps])
    rho = AMU_PER_A3_TO_GCC * mm / vpa
    full = d[cc].copy()
    full["density"] = rho
    full["vol_per_atom"] = vpa
    E = 10 ** mod["model"].predict(full[mod["features"]])
    H = hard["model"].predict(full[mod["features"]])
    fit = (W_E * np.abs(E - TARGET_E) / TARGET_E
           + W_H * np.abs(H - TARGET_H) / TARGET_H
           + W_RHO * np.abs(rho - TARGET_RHO) / TARGET_RHO)
    # bone-mineral constraint (reviewer #9): require Ca and (P or Si); penalise others
    _dd = [c.get_el_amt_dict() for c in comps]
    fit = fit + np.array([0.0 if ('Ca' in d and ('P' in d or 'Si' in d)) else 5.0 for d in _dd])
    return fit, E, H, rho


def readable(comp):
    """Reduced integer formula string, e.g. 'K4Na2MgP2O9'."""
    return comp.reduced_formula


def dedup_key(comp):
    """Distinct realistic compounds collapse by their reduced integer formula."""
    return comp.reduced_formula


class DentinDesign(Problem):
    def __init__(self):
        super().__init__(n_var=len(CATIONS), n_obj=1, xl=0.0, xu=1.0)

    def _evaluate(self, X, out, *args, **kwargs):
        fit, *_ = score_batch([decode(row) for row in X])
        out["F"] = fit.reshape(-1, 1)


seen = {}
for sd in SEEDS:
    print(f"Running GA (pymoo) for DENTIN target, seed {sd} ...")
    res = minimize(DentinDesign(),
                   GA(pop_size=60, eliminate_duplicates=True),
                   ("n_gen", 25), seed=sd, save_history=True, verbose=False)
    for algo in res.history:
        comps = [decode(r) for r in algo.pop.get("X")]
        fit, E, H, rho = score_batch(comps)
        for c, f, e, h, r in zip(comps, fit, E, H, rho):
            key = dedup_key(c)
            if key not in seen or f < seen[key][0]:
                seen[key] = (f, c, e, h, r)

rows = sorted(seen.values(), key=lambda t: t[0])[:15]
out = pd.DataFrame({
    "composition":       [readable(c) for _, c, _, _, _ in rows],
    "pred_modulus_GPa":  [round(e, 1) for _, _, e, _, _ in rows],
    "pred_hardness_GPa": [round(h, 2) for _, _, _, h, _ in rows],
    "pred_density_gcc":  [round(r, 2) for _, _, _, _, r in rows],
    "fitness":           [round(f, 4) for f, _, _, _, _ in rows],
})
out.to_csv(os.path.join(DATA, "designed_dentin_compositions.csv"), index=False)
print(f"\nGA-designed dentin-target compositions "
      f"(E={TARGET_E:.0f} GPa, H={TARGET_H:.1f} GPa, rho={TARGET_RHO:.1f} g/cm^3):")
print(out.to_string(index=False))
print("\nSaved -> designed_dentin_compositions.csv")
