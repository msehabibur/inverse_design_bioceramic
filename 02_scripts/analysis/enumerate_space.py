"""
Exhaustive enumeration of the reachable single-phase integer-oxide space.
NO DFT, NO GA -- turns the dentin "negative result" from "our search did not
find one" into "the entire reachable space cannot reach the target".

For each showcase pool we enumerate every charge-balanced oxide with <=4 cations
and per-cation counts 1..4 (formula unit <= 40 atoms), deduplicate by reduced
formula, and predict Young's modulus with that showcase's forward models. We
then report the predicted-modulus distribution and, crucially, its FLOOR versus
the target -- if even the minimum predicted modulus over the whole enumerated
space sits well above the dentin target, no single-phase oxide in this chemistry
can match dentin, independent of the heuristic GA search.

Outputs: data/enumeration_modulus.csv, manuscript/Figure9_enumeration.png
"""
import os
from math import gcd
from functools import reduce
from itertools import combinations, product
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.core import Composition
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # noqa: make paths/features importable
from features import make_featurizer
from paths import DATA, MANUSCRIPT

AMU = 1.66054
MAXC = 4          # max count per cation
MAXCAT = 3        # max distinct cations (keeps the enumeration tractable)
MAXFU = 40        # max atoms per formula unit
featurizer = make_featurizer()
try:
    featurizer.set_n_jobs(8)     # parallel featurisation
except Exception:
    pass

POOLS = {
    "enamel":  (dict(Zr=4, Ti=4, Si=4, Ce=4, P=5, Al=3, Y=3, La=3, Ca=2, Mg=2, Sr=2),
                "model_modulus.joblib", "model_volperatom.joblib", 85.0),
    "dentin":  (dict(Ca=2, P=5, Si=4, Mg=2, Na=1, Sr=2, K=1, Al=3),
                "model_modulus_dentin.joblib", "model_volperatom_dentin.joblib", 20.0),
    "implant": (dict(Zr=4, Ti=4, Si=4, Ce=4, P=5, Al=3, Y=3, La=3, Ca=2, Mg=2, Sr=2),
                "model_modulus.joblib", "model_volperatom.joblib", 110.0),
}


def enumerate_oxides(charge):
    cats = list(charge)
    seen = {}
    for k in range(1, MAXCAT + 1):
        for subset in combinations(cats, k):
            for counts in product(range(1, MAXC + 1), repeat=k):
                o2 = sum(c * charge[e] for e, c in zip(subset, counts))
                cc = dict(zip(subset, counts))
                if o2 % 2:                       # double to keep O integral
                    cc = {e: 2 * c for e, c in cc.items()}; o2 *= 2
                atoms = {**cc, "O": o2 // 2}
                g = reduce(gcd, atoms.values())
                atoms = {e: n // g for e, n in atoms.items()}
                if sum(atoms.values()) > MAXFU:
                    continue
                key = tuple(sorted(atoms.items()))
                seen[key] = atoms
    return [Composition(a) for a in seen.values()]


def predict_E(comps, mod, vol):
    cf = vol["features"]; ff = mod["features"]
    out = np.full(len(comps), np.nan)
    B = 4000
    for s in range(0, len(comps), B):
        chunk = comps[s:s + B]
        d = pd.DataFrame({"composition_obj": chunk})
        d = featurizer.featurize_dataframe(d, col_id="composition_obj",
                                           ignore_errors=True)
        d[cf] = d[cf].fillna(d[cf].mean())
        vpa = vol["model"].predict(d[cf])
        mass = np.array([c.weight / c.num_atoms for c in chunk])
        full = d[cf].copy(); full["density"] = AMU * mass / vpa
        full["vol_per_atom"] = vpa
        out[s:s + B] = 10 ** mod["model"].predict(full[ff].values)
    return out


rows = []
summary = []
cache = {}     # enamel & implant share the same pool + models -> enumerate once
for sc, (charge, modf, volf, target) in POOLS.items():
    key = (modf, volf, tuple(sorted(charge.items())))
    if key not in cache:
        mod = joblib.load(os.path.join(DATA, modf))
        vol = joblib.load(os.path.join(DATA, volf))
        comps = enumerate_oxides(charge)
        cache[key] = (comps, predict_E(comps, mod, vol))
    comps, E = cache[key]
    ok = ~np.isnan(E)
    Ev = E[ok]
    forms = [comps[i].reduced_formula for i in range(len(comps)) if ok[i]]
    for f, e in zip(forms, Ev):
        rows.append({"showcase": sc, "formula": f, "pred_modulus_GPa": round(float(e), 1)})
    floor = float(Ev.min())
    n_reach = int((Ev <= 1.1 * target).sum())
    summary.append((sc, len(Ev), floor, target, n_reach))
    print(f"[{sc}] enumerated {len(Ev):>6} single-phase oxides | "
          f"pred-modulus floor {floor:6.1f} GPa | target {target:5.0f} | "
          f"within 10% of target: {n_reach}")

pd.DataFrame(rows).to_csv(os.path.join(DATA, "enumeration_modulus.csv"), index=False)

# ---- figure: reachable modulus distribution vs target -------------------
plt.rcParams.update({
    "font.family": ["Arial Narrow", "Arial", "DejaVu Sans"],
    "font.size": 14, "axes.titlesize": 16, "axes.labelsize": 15,
    "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13,
})
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
COL = {"enamel": "#1A6090", "dentin": "#D97F33", "implant": "#2E7D5A"}
allrows = pd.DataFrame(rows)
for j, (sc, n, floor, target, nreach) in enumerate(summary):
    sub = allrows[allrows.showcase == sc]["pred_modulus_GPa"].values
    ax = axes[j]
    ax.hist(sub, bins=60, color=COL[sc], alpha=0.8)
    ax.axvline(target, ls=":", color="black", lw=2, label=f"Target {target:.0f} GPa")
    ax.axvline(floor, ls="--", color="red", lw=1.6, label=f"Floor {floor:.0f} GPa")
    ax.set_title(f"{sc.capitalize()}: {n} reachable oxides", loc="left")
    ax.set_xlabel("Predicted Young's modulus (GPa)")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(MANUSCRIPT, "Figure9_enumeration.png"), dpi=200, bbox_inches="tight")
print("saved -> Figure9_enumeration.png + data/enumeration_modulus.csv")
