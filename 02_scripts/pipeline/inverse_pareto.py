"""
Weight-free multi-objective check (NSGA-II) for each showcase.
NO DFT. Addresses the "arbitrary fitness weights" criticism: instead of a single
weighted-sum objective, we run NSGA-II on two objectives -- distance to the
target modulus and distance to the target density -- and report the Pareto front.
If the single-objective weighted-sum design sits on/near this front, the result
is not an artefact of the chosen weights.

Outputs: data/pareto_<showcase>.csv, manuscript/figure_pareto.png
"""
import os
from math import gcd
from functools import reduce
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.core import Composition
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # noqa: make paths/features importable
from features import make_featurizer
from paths import DATA, MANUSCRIPT, MODELS

AMU = 1.66054
MAX_COUNT = 6
featurizer = make_featurizer()
try:
    featurizer.set_n_jobs(8)
except Exception:
    pass

SHOWCASES = {
    "enamel":  (dict(Zr=4, Ti=4, Si=4, Ce=4, P=5, Al=3, Y=3, La=3, Ca=2, Mg=2, Sr=2),
                "model_modulus.joblib", "model_volperatom.joblib", 85.0, 3.0),
    "dentin":  (dict(Ca=2, P=5, Si=4, Mg=2, Na=1, Sr=2, K=1, Al=3),
                "model_modulus_dentin.joblib", "model_volperatom_dentin.joblib", 20.0, 2.1),
    "implant": (dict(Zr=4, Ti=4, Si=4, Ce=4, P=5, Al=3, Y=3, La=3, Ca=2, Mg=2, Sr=2),
                "model_modulus.joblib", "model_volperatom.joblib", 110.0, 4.4),
}


def decode(x, charge, cats):
    amt = dict(zip(cats, x))
    top = sorted(amt.items(), key=lambda kv: -kv[1])[:4]
    keep = {e: a for e, a in top if a > 0.10} or {cats[0]: 1.0}
    lo = min(keep.values())
    counts = {e: max(1, min(MAX_COUNT, int(round(a / lo)))) for e, a in keep.items()}
    o2 = sum(n * charge[e] for e, n in counts.items())
    if o2 % 2:
        counts = {e: 2 * n for e, n in counts.items()}; o2 *= 2
    atoms = {**counts, "O": o2 // 2}
    g = reduce(gcd, atoms.values())
    return Composition({e: n // g for e, n in atoms.items()})


def forward(comps, mod, vol):
    cf = vol["features"]; ff = mod["features"]
    d = pd.DataFrame({"composition_obj": comps})
    d = featurizer.featurize_dataframe(d, col_id="composition_obj", ignore_errors=True)
    d[cf] = d[cf].fillna(d[cf].mean())
    vpa = vol["model"].predict(d[cf])
    mass = np.array([c.weight / c.num_atoms for c in comps])
    rho = AMU * mass / vpa
    full = d[cf].copy(); full["density"] = rho; full["vol_per_atom"] = vpa
    E = 10 ** mod["model"].predict(full[ff].values)
    return E, rho


fig, axes = plt.subplots(1, 3, figsize=(16, 5))
COL = {"enamel": "#1A6090", "dentin": "#D97F33", "implant": "#2E7D5A"}

for j, (sc, (charge, modf, volf, tE, tR)) in enumerate(SHOWCASES.items()):
    mod = joblib.load(os.path.join(MODELS, modf))
    vol = joblib.load(os.path.join(MODELS, volf))
    cats = list(charge)

    class P(Problem):
        def __init__(self):
            super().__init__(n_var=len(cats), n_obj=2, xl=0.0, xu=1.0)
        def _evaluate(self, X, out, *a, **k):
            comps = [decode(r, charge, cats) for r in X]
            E, rho = forward(comps, mod, vol)
            out["F"] = np.column_stack([np.abs(E - tE) / tE, np.abs(rho - tR) / tR])

    res = minimize(P(), NSGA2(pop_size=80), ("n_gen", 40), seed=42, verbose=False)
    comps = [decode(r, charge, cats) for r in res.X]
    E, rho = forward(comps, mod, vol)
    df = pd.DataFrame({"formula": [c.reduced_formula for c in comps],
                       "pred_modulus_GPa": np.round(E, 1),
                       "pred_density_gcc": np.round(rho, 2),
                       "dist_E": np.round(res.F[:, 0], 4),
                       "dist_rho": np.round(res.F[:, 1], 4)}).drop_duplicates("formula")
    df = df.sort_values("dist_E")
    df.to_csv(os.path.join(DATA, f"pareto_{sc}.csv"), index=False)
    print(f"[{sc}] Pareto front: {len(df)} non-dominated designs; "
          f"modulus span {E.min():.0f}-{E.max():.0f} GPa (target {tE:.0f})")

    ax = axes[j]
    ax.scatter(E, rho, c=COL[sc], s=40, alpha=0.8, label="Pareto front")
    ax.scatter([tE], [tR], marker="*", s=320, color="black", zorder=5, label="target")
    ax.set_xlabel("predicted modulus (GPa)")
    ax.set_ylabel("predicted density (g/cm$^3$)")
    ax.set_title(f"{sc.capitalize()} — NSGA-II front", loc="left")
    ax.legend(frameon=False, fontsize=9)

fig.tight_layout()
fig.savefig(os.path.join(MANUSCRIPT, "figure_pareto.png"), dpi=200, bbox_inches="tight")
print("saved -> figure_pareto.png + data/pareto_*.csv")
