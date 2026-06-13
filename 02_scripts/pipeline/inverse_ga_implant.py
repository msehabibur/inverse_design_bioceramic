"""
Showcase 4 - Step 3.  Genetic-algorithm inverse design targeting the
mechanical fingerprint of the Ti--6Al--4V dental implant alloy.

Targets:  Young's modulus 110 GPa, Vickers hardness 3.5 GPa, density 4.4 g/cm^3.

Clinical motivation: the long-term success of an osseointegrated implant is
limited by stress shielding at the implant-bone interface, which arises from
the modulus mismatch between Ti-6Al-4V (~110 GPa) and cortical bone (~15 GPa).
A bioceramic coating or interface layer whose intrinsic stiffness matches the
Ti-6Al-4V substrate would reduce this mismatch on the metal side.  This
showcase asks the genetic algorithm to inversely design an oxide composition
matching Ti-6Al-4V's mechanical fingerprint.

Same GA architecture as inverse_ga.py: pymoo single-objective, charge-balanced
oxide decoding, full enamel-style biocompatible cation pool.

NOTE: thermodynamic stability is NOT enforced; designed candidates need a
downstream stability check (Showcase 5, scripts/stability_check.py).

Output: designed_implant_compositions.csv
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

# --- design targets: Ti-6Al-4V mechanical fingerprint ---
TARGET_E, TARGET_H, TARGET_RHO = 110.0, 3.5, 4.4
W_E, W_H, W_RHO = 0.50, 0.15, 0.35    # same weights as enamel/dentin for comparability

# --- biocompatible cation pool (same as enamel showcase) ---
CHARGE = {"Zr": 4, "Ti": 4, "Si": 4, "Ce": 4, "P": 5,
          "Al": 3, "Y": 3, "La": 3, "Ca": 2, "Mg": 2, "Sr": 2}
CATIONS = list(CHARGE)

MAX_COUNT = 6      # largest count of any single cation in a formula unit
SEEDS = [42, 1, 7, 13, 21]   # repeat the search; keep best per distinct formula

mod  = joblib.load(os.path.join(DATA, "model_modulus.joblib"))
hard = joblib.load(os.path.join(DATA, "model_hardness.joblib"))
vol  = joblib.load(os.path.join(DATA, "model_volperatom.joblib"))
featurizer = make_featurizer()
AMU_PER_A3_TO_GCC = 1.66054


def decode(x):
    """GA genome -> small-integer, charge-balanced oxide (see inverse_ga.py)."""
    amt = dict(zip(CATIONS, x))
    top = sorted(amt.items(), key=lambda kv: -kv[1])[:4]
    keep = {e: a for e, a in top if a > 0.10}
    if not keep:
        keep = {"Zr": 1.0}
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
    return fit, E, H, rho


def readable(comp):
    """Reduced integer formula string, e.g. 'Sr2Ce2Ti2P2O15'."""
    return comp.reduced_formula


def dedup_key(comp):
    """Distinct realistic compounds collapse by their reduced integer formula."""
    return comp.reduced_formula


class ImplantDesign(Problem):
    def __init__(self):
        super().__init__(n_var=len(CATIONS), n_obj=1, xl=0.0, xu=1.0)

    def _evaluate(self, X, out, *args, **kwargs):
        fit, *_ = score_batch([decode(row) for row in X])
        out["F"] = fit.reshape(-1, 1)


seen = {}
for sd in SEEDS:
    print(f"Running GA (pymoo) for Ti-6Al-4V interface target, seed {sd} ...")
    res = minimize(ImplantDesign(),
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
out.to_csv(os.path.join(DATA, "designed_implant_compositions.csv"), index=False)
print(f"\nGA-designed implant-interface compositions "
      f"(E={TARGET_E:.0f} GPa, H={TARGET_H:.1f} GPa, rho={TARGET_RHO:.1f} g/cm^3):")
print(out.to_string(index=False))
print("\nSaved -> designed_implant_compositions.csv")
