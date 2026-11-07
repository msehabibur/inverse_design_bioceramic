"""Re-run the search for all three case studies with the per-generation history kept.

The submitted convergence panels plotted one seed, which is why the referees could not see
the stabilisation the caption claimed. The search runs from five seeds and pools the
designs, so the diagnostic has to be the across-seed curve. This records the best objective
value at every generation for every seed, and the final population in the plane of
predicted modulus and predicted density.

Writes 01_data/revision_ga_history.csv and 01_data/revision_ga_population.csv.
"""
import os
import sys
from functools import reduce
from math import gcd
import numpy as np
import pandas as pd
import joblib
from pymatgen.core import Composition
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features import make_featurizer
from paths import DATA

AMU = 1.66054
MAX_COUNT = 6
SEEDS = [42, 1, 7, 13, 21]
POP, NGEN = 60, 25
W_E, W_H, W_RHO = 0.588, 0.0, 0.412

POOL_MAIN = {"Zr": 4, "Ti": 4, "Si": 4, "Ce": 4, "P": 5,
             "Al": 3, "Y": 3, "La": 3, "Ca": 2, "Mg": 2, "Sr": 2}
POOL_DENTIN = {"Ca": 2, "P": 5, "Si": 4, "Mg": 2, "Na": 1, "Sr": 2, "K": 1, "Al": 3}

CASES = {
    "enamel": dict(charge=POOL_MAIN, target=(85.0, 4.0, 3.0), constraint=False,
                   models=("model_modulus.joblib", "model_hardness.joblib",
                           "model_volperatom.joblib")),
    "dentin": dict(charge=POOL_DENTIN, target=(20.0, 0.7, 2.1), constraint=True,
                   models=("model_modulus_dentin.joblib",
                           "model_hardness_dentin.joblib",
                           "model_volperatom_dentin.joblib")),
    "implant": dict(charge=POOL_MAIN, target=(110.0, 3.5, 4.4), constraint=False,
                    models=("model_modulus.joblib", "model_hardness.joblib",
                            "model_volperatom.joblib")),
}
FEAT = make_featurizer()


def decode(x, charge):
    cations = list(charge)
    amt = dict(zip(cations, x))
    top = sorted(amt.items(), key=lambda kv: -kv[1])[:4]
    keep = {e: a for e, a in top if a > 0.10}
    if not keep:
        keep = {cations[0]: 1.0}
    lo = min(keep.values())
    counts = {e: max(1, min(MAX_COUNT, int(round(a / lo)))) for e, a in keep.items()}
    o2 = sum(n * charge[e] for e, n in counts.items())
    if o2 % 2:
        counts = {e: 2 * n for e, n in counts.items()}
        o2 *= 2
    atoms = {**counts, "O": o2 // 2}
    g = reduce(gcd, atoms.values())
    return Composition({e: n // g for e, n in atoms.items()})


class Scorer:
    def __init__(self, case):
        cfg = CASES[case]
        self.charge = cfg["charge"]
        self.tE, self.tH, self.tR = cfg["target"]
        self.constraint = cfg["constraint"]
        self.mod = joblib.load(os.path.join(DATA, cfg["models"][0]))
        self.hard = joblib.load(os.path.join(DATA, cfg["models"][1]))
        self.vol = joblib.load(os.path.join(DATA, cfg["models"][2]))

    def __call__(self, comps):
        d = pd.DataFrame({"composition_obj": comps})
        d = FEAT.featurize_dataframe(d, col_id="composition_obj", ignore_errors=True)
        cc = self.vol["features"]
        d[cc] = d[cc].fillna(d[cc].mean())
        vpa = np.clip(self.vol["model"].predict(d[cc]), 1e-3, None)
        mm = np.array([c.weight / c.num_atoms for c in comps])
        rho = AMU * mm / vpa
        full = d[cc].copy()
        full["density"] = rho
        full["vol_per_atom"] = vpa
        E = 10 ** self.mod["model"].predict(full[self.mod["features"]])
        H = self.hard["model"].predict(full[self.mod["features"]])
        f = (W_E * np.abs(E - self.tE) / self.tE
             + W_H * np.abs(H - self.tH) / self.tH
             + W_RHO * np.abs(rho - self.tR) / self.tR)
        if self.constraint:
            dd = [c.get_el_amt_dict() for c in comps]
            f = f + np.array([0.0 if ("Ca" in x and ("P" in x or "Si" in x)) else 5.0
                              for x in dd])
        return f, E, H, rho


class Design(Problem):
    def __init__(self, sc):
        self.sc = sc
        super().__init__(n_var=len(sc.charge), n_obj=1, xl=0.0, xu=1.0)

    def _evaluate(self, X, out, *a, **k):
        f, *_ = self.sc([decode(r, self.sc.charge) for r in X])
        out["F"] = f.reshape(-1, 1)


hist, popn = [], []
for case in CASES:
    sc = Scorer(case)
    print(f"#### {case}", flush=True)
    for sd in SEEDS:
        res = minimize(Design(sc), GA(pop_size=POP, eliminate_duplicates=True),
                       ("n_gen", NGEN), seed=sd, save_history=True, verbose=False)
        run = []
        for g, algo in enumerate(res.history, start=1):
            run.append(float(algo.pop.get("F").min()))
            hist.append(dict(case=case, seed=sd, generation=g,
                             best_fitness=run[-1],
                             running_best=float(np.min(run))))
        last = res.history[-1]
        comps = [decode(r, sc.charge) for r in last.pop.get("X")]
        f, E, H, rho = sc(comps)
        for c, fi, ei, ri in zip(comps, f, E, rho):
            popn.append(dict(case=case, seed=sd, composition=c.reduced_formula,
                             fitness=float(fi), pred_E_GPa=float(ei),
                             pred_rho_gcc=float(ri)))
        print(f"   seed {sd}: best {min(run):.4f} at generation "
              f"{int(np.argmin(run)) + 1}", flush=True)

h = pd.DataFrame(hist)
h.to_csv(os.path.join(DATA, "revision_ga_history.csv"), index=False)
pd.DataFrame(popn).to_csv(os.path.join(DATA, "revision_ga_population.csv"), index=False)
print("\nwrote 01_data/revision_ga_history.csv and revision_ga_population.csv", flush=True)

for case in CASES:
    s = h[h.case == case]
    per_seed = s.groupby("seed").running_best
    last_imp = []
    for sd, g in s.groupby("seed"):
        rb = g.sort_values("generation").running_best.values
        last_imp.append(int(np.max(np.nonzero(np.diff(rb, prepend=rb[0] + 1))[0]) + 1))
    pooled = s.groupby("generation").running_best.min().values
    pl = int(np.max(np.nonzero(np.diff(pooled, prepend=pooled[0] + 1))[0]) + 1)
    print(f"  {case:8s} pooled best stops improving at generation {pl} of {NGEN}; "
          f"per-seed last improvement {min(last_imp)} to {max(last_imp)}", flush=True)

print("\nREVISION_GA_HISTORY_DONE", flush=True)
