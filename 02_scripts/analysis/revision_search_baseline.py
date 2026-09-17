"""Revision analysis R3 - is the genetic algorithm doing any work, and does the
enumeration actually cover the space the genetic algorithm searches?

Answers reviewer 3 point 3 and repairs the enumeration-scope gap reviewer 1
point 6 points at:

  * random-search control. The genetic algorithm spends 5 seeds x 60 individuals
    x 25 generations = 7,500 objective evaluations. We draw the same 7,500
    genomes uniformly at random, decode and score them with the identical
    objective, repeat that ten times for an error bar, and report the best
    objective reached by each search. We also record how many random draws are
    needed to match the genetic algorithm's best value.
  * enumeration at the search bounds. The published enumeration allowed at most
    3 distinct cations with counts 1-4, whereas the decoder emits up to 4
    distinct cations with counts up to 6, doubled to 12 when oxygen parity
    requires it. Only 2 of the 45 published candidates lie inside the old
    enumeration. We re-enumerate at the decoder's own bounds and recompute the
    modulus floor, so the exhaustive statement covers the space actually
    searched.

Writes 01_data/revision_search_baseline.csv, 01_data/revision_enumeration.csv
and 01_data/revision_enumeration_full.csv.
"""
import os
import sys
import itertools
from functools import reduce
from math import gcd
import numpy as np
import pandas as pd
import joblib
from pymatgen.core import Composition

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features import make_featurizer
from paths import DATA

AMU = 1.66054
MAX_COUNT = 6
MAX_CATIONS = 4
# No cap on the formula unit. A 60-atom limit was applied here and never
# declared, and the decoder has no such limit: the parity doubling alone can
# carry a reduced formula past 60 atoms, so a capped enumeration is not the
# decoder's image and cannot support an exhaustiveness claim.
MAXFU = 10 ** 9
GA_BUDGET = 5 * 60 * 25          # seeds x population x generations
RANDOM_REPEATS = 10

POOL_MAIN = {"Zr": 4, "Ti": 4, "Si": 4, "Ce": 4, "P": 5,
             "Al": 3, "Y": 3, "La": 3, "Ca": 2, "Mg": 2, "Sr": 2}
POOL_DENTIN = {"Ca": 2, "P": 5, "Si": 4, "Mg": 2,
               "Na": 1, "Sr": 2, "K": 1, "Al": 3}

CASES = {
    "enamel":  dict(charge=POOL_MAIN, target=(85.0, 4.0, 3.0),
                    models=("model_modulus.joblib", "model_hardness.joblib",
                            "model_volperatom.joblib"), constraint=False),
    "dentin":  dict(charge=POOL_DENTIN, target=(20.0, 0.7, 2.1),
                    models=("model_modulus_dentin.joblib",
                            "model_hardness_dentin.joblib",
                            "model_volperatom_dentin.joblib"), constraint=True),
    "implant": dict(charge=POOL_MAIN, target=(110.0, 3.5, 4.4),
                    models=("model_modulus.joblib", "model_hardness.joblib",
                            "model_volperatom.joblib"), constraint=False),
}
W_E, W_H, W_RHO = 0.588, 0.0, 0.412

FEAT = make_featurizer()


def decode(x, charge):
    """Identical to the decoder in inverse_ga*.py."""
    cations = list(charge)
    amt = dict(zip(cations, x))
    top = sorted(amt.items(), key=lambda kv: -kv[1])[:MAX_CATIONS]
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
    atoms = {e: n // g for e, n in atoms.items()}
    return Composition(atoms)


class Scorer:
    """Design-time chain and objective, identical to the genetic algorithm."""

    def __init__(self, case):
        cfg = CASES[case]
        self.charge = cfg["charge"]
        self.tE, self.tH, self.tR = cfg["target"]
        self.constraint = cfg["constraint"]
        self.mod = joblib.load(os.path.join(DATA, cfg["models"][0]))
        self.hard = joblib.load(os.path.join(DATA, cfg["models"][1]))
        self.vol = joblib.load(os.path.join(DATA, cfg["models"][2]))

    def __call__(self, comps, chunk=4000):
        E, H, R, F = [], [], [], []
        for i in range(0, len(comps), chunk):
            part = comps[i:i + chunk]
            d = pd.DataFrame({"composition_obj": part})
            d = FEAT.featurize_dataframe(d, col_id="composition_obj",
                                         ignore_errors=True)
            cc = self.vol["features"]
            d[cc] = d[cc].fillna(d[cc].mean())
            vpa = self.vol["model"].predict(d[cc])
            mm = np.array([c.weight / c.num_atoms for c in part])
            rho = AMU * mm / vpa
            full = d[cc].copy()
            full["density"] = rho
            full["vol_per_atom"] = vpa
            e = 10 ** self.mod["model"].predict(full[self.mod["features"]])
            h = self.hard["model"].predict(full[self.mod["features"]])
            f = (W_E * np.abs(e - self.tE) / self.tE
                 + W_H * np.abs(h - self.tH) / self.tH
                 + W_RHO * np.abs(rho - self.tR) / self.tR)
            if self.constraint:
                dd = [c.get_el_amt_dict() for c in part]
                f = f + np.array([0.0 if ("Ca" in x and ("P" in x or "Si" in x))
                                  else 5.0 for x in dd])
            E.append(e); H.append(h); R.append(rho); F.append(f)
            print(f"      scored {min(i + chunk, len(comps))}/{len(comps)}",
                  flush=True)
        return (np.concatenate(F), np.concatenate(E),
                np.concatenate(H), np.concatenate(R))


# ------------------------------------------------- part A: random control
base = []
for case in CASES:
    print(f"\n#### random-search control, {case}", flush=True)
    sc = Scorer(case)
    n_var = len(sc.charge)
    ga = pd.read_csv(os.path.join(DATA, {
        "enamel": "designed_compositions.csv",
        "dentin": "designed_dentin_compositions.csv",
        "implant": "designed_implant_compositions.csv"}[case]))
    ga_best_fit = float(ga["fitness"].min())
    ga_best_E = float(ga.loc[ga["fitness"].idxmin(), "pred_modulus_GPa"])

    for rep in range(RANDOM_REPEATS):
        rng = np.random.default_rng(1000 + rep)
        X = rng.random((GA_BUDGET, n_var))
        comps = [decode(row, sc.charge) for row in X]
        uniq = {}
        for c in comps:
            uniq.setdefault(c.reduced_formula, c)
        keys = list(uniq)
        fit, E, H, R = sc([uniq[k] for k in keys])
        order = np.argsort(fit)
        rank_hit = int(np.searchsorted(np.sort(fit), ga_best_fit))
        base.append(dict(case=case, search="random", repeat=rep,
                         evaluations=GA_BUDGET,
                         distinct_formulas=len(keys),
                         best_fitness=round(float(fit[order[0]]), 4),
                         best_E_GPa=round(float(E[order[0]]), 1),
                         ga_best_fitness=round(ga_best_fit, 4),
                         ga_best_E_GPa=round(ga_best_E, 1),
                         n_random_better_than_ga=rank_hit))
        print(f"   repeat {rep}: {len(keys)} distinct formulas, "
              f"random best fitness {fit[order[0]]:.4f} "
              f"(E {E[order[0]]:.1f}) vs GA {ga_best_fit:.4f} "
              f"(E {ga_best_E:.1f}); random draws beating GA: {rank_hit}",
              flush=True)

bl = pd.DataFrame(base)
bl.to_csv(os.path.join(DATA, "revision_search_baseline.csv"), index=False)
print("\nwrote 01_data/revision_search_baseline.csv", flush=True)
for case in CASES:
    s = bl[bl.case == case]
    print(f"  {case:8s} GA {s.ga_best_fitness.iloc[0]:.4f} | random over "
          f"{RANDOM_REPEATS} repeats {s.best_fitness.mean():.4f} "
          f"+/- {s.best_fitness.std():.4f} "
          f"(best {s.best_fitness.min():.4f}) | random draws better than GA: "
          f"{s.n_random_better_than_ga.mean():.1f} of "
          f"{s.distinct_formulas.mean():.0f}", flush=True)

# ------------------------------ part B: enumeration over the decoder's image
# Every count vector the decoder can produce has a minimum of one, because the
# least-abundant retained cation is normalised to one before the cap at
# MAX_COUNT. Enumerating over vectors with a higher minimum as well would give a
# superset whose minimum prediction is no higher, so the floors reported here are
# floors over what the decoder can actually emit. revision_decoder_image.py
# measures the difference.
enum_rows, floors = [], []
for case in ("enamel", "dentin"):        # implant shares the enamel pool
    print(f"\n#### enumeration at decoder bounds, {case} pool", flush=True)
    sc = Scorer(case)
    charge = sc.charge
    cations = list(charge)
    seen = {}
    for k in range(1, MAX_CATIONS + 1):
        for subset in itertools.combinations(cations, k):
            for counts in itertools.product(range(1, MAX_COUNT + 1), repeat=k):
                if min(counts) != 1:
                    # decode() sets the least-abundant retained cation to one,
                    # so vectors with a higher minimum are outside its image
                    continue
                cc = dict(zip(subset, counts))
                o2 = sum(n * charge[e] for e, n in cc.items())
                if o2 % 2:
                    cc = {e: 2 * n for e, n in cc.items()}
                    o2 *= 2
                atoms = {**cc, "O": o2 // 2}
                if sum(atoms.values()) > MAXFU:
                    continue
                g = reduce(gcd, atoms.values())
                atoms = {e: n // g for e, n in atoms.items()}
                c = Composition(atoms)
                if sc.constraint:
                    x = c.get_el_amt_dict()
                    if not ("Ca" in x and ("P" in x or "Si" in x)):
                        continue
                seen.setdefault(c.reduced_formula, c)
    keys = list(seen)
    print(f"   {len(keys)} distinct charge-balanced reduced formulas in the decoder's "
          f"image", flush=True)
    fit, E, H, R = sc([seen[k] for k in keys])
    df = pd.DataFrame(dict(pool=case, composition=keys,
                           pred_modulus_GPa=np.round(E, 2),
                           pred_density_gcc=np.round(R, 3),
                           fitness=np.round(fit, 4)))
    enum_rows.append(df)
    floors.append(dict(pool=case, n_formulas=len(keys),
                       modulus_floor_GPa=round(float(E.min()), 1),
                       floor_composition=keys[int(np.argmin(E))],
                       modulus_p01_GPa=round(float(np.percentile(E, 1)), 1),
                       modulus_median_GPa=round(float(np.median(E)), 1),
                       modulus_max_GPa=round(float(E.max()), 1)))
    print(f"   modulus floor {E.min():.1f} GPa at {keys[int(np.argmin(E))]}",
          flush=True)

full = pd.concat(enum_rows, ignore_index=True)
full.to_csv(os.path.join(DATA, "revision_enumeration_full.csv"), index=False)
fl = pd.DataFrame(floors)
fl.to_csv(os.path.join(DATA, "revision_enumeration.csv"), index=False)
print("\nwrote 01_data/revision_enumeration.csv and _full.csv", flush=True)
print(fl.to_string(index=False), flush=True)

# do the 45 published candidates now lie inside the enumerated space?
for name, fname in (("enamel", "designed_compositions.csv"),
                    ("dentin", "designed_dentin_compositions.csv"),
                    ("implant", "designed_implant_compositions.csv")):
    ga = pd.read_csv(os.path.join(DATA, fname))
    pool = "dentin" if name == "dentin" else "enamel"
    inside = set(full[full.pool == pool]["composition"])
    hit = sum(1 for c in ga["composition"]
              if Composition(c).reduced_formula in inside)
    print(f"  {name:8s}: {hit}/{len(ga)} published candidates inside the "
          f"re-enumerated space", flush=True)

print("\nREVISION_BASELINE_DONE", flush=True)
