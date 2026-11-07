"""
Step 4 - Real inverse design: a genetic algorithm that designs dental
ceramics matching enamel's mechanical fingerprint.

Targets:  Young's modulus 85 GPa, Vickers hardness 4 GPa, density 3.0 g/cm^3.

The GA (pymoo) proposes cation mixtures; each candidate is snapped to a SMALL
INTEGER formula unit (least-abundant cation -> 1, capped at MAX_COUNT) and the
oxygen content is fixed by CHARGE BALANCE, so every candidate is a realistic,
weighable single-phase oxide stoichiometry (e.g. SrMgSiP2O9) rather than a
continuous point in composition space. For each candidate the vol-per-atom
helper estimates structure, density follows exactly from composition +
vol/atom, and the modulus & hardness models predict the mechanical properties.
Fitness = weighted distance to the three targets. The search is repeated over
several random seeds and the best result per distinct reduced formula is kept.

NOTE: thermodynamic stability is NOT enforced -- composition-only stability
prediction is unreliable -- so designed candidates need a downstream DFT/MLFF
stability check before being taken seriously.

Output: designed_compositions.csv
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
from paths import DATA
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize

# --- design targets: enamel mechanical fingerprint ---
TARGET_E, TARGET_H, TARGET_RHO = 85.0, 4.0, 3.0      # GPa, GPa, g/cm^3
# fitness weights -- trust modulus & density more than the weak hardness model
W_E, W_H, W_RHO = 0.588, 0.0, 0.412   # hardness EXCLUDED from objective (weak R2, collinear with E)

# --- biocompatible cations + their typical oxidation states ---
CHARGE = {"Zr": 4, "Ti": 4, "Si": 4, "Ce": 4, "P": 5,
          "Al": 3, "Y": 3, "La": 3, "Ca": 2, "Mg": 2, "Sr": 2}
CATIONS = list(CHARGE)

MAX_COUNT = 6      # largest count of any single cation in a formula unit
SEEDS = [42, 1, 7, 13, 21]   # repeat the search; keep best per distinct formula

mod = joblib.load(os.path.join(DATA, "model_modulus.joblib"))
hard = joblib.load(os.path.join(DATA, "model_hardness.joblib"))
vol = joblib.load(os.path.join(DATA, "model_volperatom.joblib"))
featurizer = make_featurizer()
AMU_PER_A3_TO_GCC = 1.66054      # (amu / A^3) -> g/cm^3


def decode(x):
    """GA genome (amount per cation) -> a small-integer, charge-balanced oxide.

    The <=4 most abundant cations are snapped to a small integer formula unit
    (least-abundant cation -> 1, each capped at MAX_COUNT), oxygen is fixed by
    charge balance, and the whole formula is reduced to lowest terms. The result
    is a realistic weighable stoichiometry (e.g. SrMgSiP2O9), not a continuous
    composition.
    """
    amt = dict(zip(CATIONS, x))
    top = sorted(amt.items(), key=lambda kv: -kv[1])[:4]   # keep <=4 cations
    keep = {e: a for e, a in top if a > 0.10}
    if not keep:                                            # degenerate genome
        keep = {"Zr": 1.0}
    lo = min(keep.values())
    counts = {e: max(1, min(MAX_COUNT, int(round(a / lo)))) for e, a in keep.items()}
    o2 = sum(n * CHARGE[e] for e, n in counts.items())      # 2 * oxygen
    if o2 % 2:                                              # odd -> double so O integral
        counts = {e: 2 * n for e, n in counts.items()}
        o2 *= 2
    atoms = {**counts, "O": o2 // 2}
    g = reduce(gcd, atoms.values())                         # reduce to lowest terms
    atoms = {e: n // g for e, n in atoms.items()}
    return Composition(atoms)


def score_batch(comps):
    """Predict E, H, rho for a list of compositions; return fitness + props."""
    df = pd.DataFrame({"composition_obj": comps})
    df = featurizer.featurize_dataframe(df, col_id="composition_obj",
                                        ignore_errors=True)
    cc = vol["features"]                              # composition-only features
    df[cc] = df[cc].fillna(df[cc].mean())             # guard stray NaNs
    vpa = vol["model"].predict(df[cc])                # predicted volume/atom
    mean_mass = np.array([c.weight / c.num_atoms for c in comps])
    rho = AMU_PER_A3_TO_GCC * mean_mass / vpa         # density follows exactly
    full = df[cc].copy()
    full["density"] = rho
    full["vol_per_atom"] = vpa
    fc = mod["features"]
    E = 10 ** mod["model"].predict(full[fc])
    H = hard["model"].predict(full[fc])
    fit = (W_E * np.abs(E - TARGET_E) / TARGET_E
           + W_H * np.abs(H - TARGET_H) / TARGET_H
           + W_RHO * np.abs(rho - TARGET_RHO) / TARGET_RHO)
    return fit, E, H, rho


def readable(comp):
    """Reduced integer formula string, e.g. 'Sr2Mg2Si2P2O13'."""
    return comp.reduced_formula


def dedup_key(comp):
    """Distinct realistic compounds collapse by their reduced integer formula."""
    return comp.reduced_formula


class EnamelDesign(Problem):
    def __init__(self):
        super().__init__(n_var=len(CATIONS), n_obj=1, xl=0.0, xu=1.0)

    def _evaluate(self, X, out, *args, **kwargs):
        fit, *_ = score_batch([decode(row) for row in X])
        out["F"] = fit.reshape(-1, 1)


# harvest every composition evaluated across all seeds; keep best per formula
seen = {}
for sd in SEEDS:
    print(f"Running the genetic algorithm (pymoo), seed {sd} ...")
    res = minimize(EnamelDesign(),
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
out.to_csv(os.path.join(DATA, "designed_compositions.csv"), index=False)
print(f"\nGA-designed oxide compositions closest to enamel "
      f"(E={TARGET_E:.0f} GPa, H={TARGET_H:.0f} GPa, rho={TARGET_RHO:.1f} g/cm^3):")
print(out.to_string(index=False))
print("\nSaved -> designed_compositions.csv")
