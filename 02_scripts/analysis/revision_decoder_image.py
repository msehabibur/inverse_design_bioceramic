"""The decoder's exact image, and whether the enumeration matches it.

The published decoder normalises the least-abundant retained cation to one, caps
every count at MAX_COUNT, doubles all counts when the oxygen charge comes out
odd, and reduces the formula to lowest terms. Its image is therefore finite and
can be constructed exactly: take every subset of at most MAX_CATIONS cations and
every count vector in 1..MAX_COUNT whose minimum is one, then push it through the
same parity and reduction steps. Any such vector is attainable, because the
genome amounts are continuous and only their ratios matter.

The enumeration used in the revision dropped the minimum-count-of-one condition,
so it is a superset of the image. This script measures how much larger it is,
which formulas of the enumeration are unreachable, and what the modulus floor is
over the image alone.

Writes 01_data/audit_decoder_image.csv and 01_data/audit_decoder_reachability.csv.
"""
import os
import sys
import itertools
from functools import reduce
from math import gcd

import numpy as np
import pandas as pd
from pymatgen.core import Composition

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
from paths import DATA
import importlib.util

spec = importlib.util.spec_from_file_location(
    "_rsb", os.path.join(HERE, "revision_search_baseline.py"))
# the baseline script runs its campaign at import, so lift the pieces instead
src = open(os.path.join(HERE, "revision_search_baseline.py")).read()
head = src[:src.index("# ------------------------------------------------- part A")]
ns = {"__name__": "_rsb_head", "__file__": os.path.join(HERE, "revision_search_baseline.py")}
exec(compile(head, "revision_search_baseline_head", "exec"), ns)
Scorer = ns["Scorer"]
MAX_COUNT = ns["MAX_COUNT"]
MAX_CATIONS = ns["MAX_CATIONS"]
MAXFU = ns["MAXFU"]


def reduce_formula(counts, charge):
    """The decoder's own parity and reduction steps."""
    o2 = sum(n * charge[e] for e, n in counts.items())
    cc = dict(counts)
    if o2 % 2:
        cc = {e: 2 * n for e, n in cc.items()}
        o2 *= 2
    atoms = {**cc, "O": o2 // 2}
    if sum(atoms.values()) > MAXFU:
        return None
    g = reduce(gcd, atoms.values())
    return Composition({e: n // g for e, n in atoms.items()})


rows, reach = [], []
for case in ("enamel", "dentin"):
    sc = Scorer(case)
    charge = sc.charge
    cations = list(charge)
    image, superset = {}, {}
    for k in range(1, MAX_CATIONS + 1):
        for subset in itertools.combinations(cations, k):
            for counts in itertools.product(range(1, MAX_COUNT + 1), repeat=k):
                cc = dict(zip(subset, counts))
                c = reduce_formula(cc, charge)
                if c is None:
                    continue
                if sc.constraint:
                    x = c.get_el_amt_dict()
                    if not ("Ca" in x and ("P" in x or "Si" in x)):
                        continue
                superset.setdefault(c.reduced_formula, c)
                if min(counts) == 1:            # the decoder normalises to one
                    image.setdefault(c.reduced_formula, c)
    keys = list(image)
    print(f"\n#### {case} pool", flush=True)
    print(f"  enumeration without the normalisation condition : {len(superset)}",
          flush=True)
    print(f"  the decoder's exact image                        : {len(keys)}",
          flush=True)
    print(f"  enumerated formulas the decoder cannot emit      : "
          f"{len(superset) - len(keys)}", flush=True)
    fit, E, H, R = sc([image[q] for q in keys])
    i = int(np.argmin(E))
    print(f"  modulus floor over the image : {E.min():.2f} GPa at {keys[i]}", flush=True)
    rows.append(dict(pool=case, n_superset=len(superset), n_image=len(keys),
                     n_unreachable=len(superset) - len(keys),
                     image_floor_GPa=round(float(E.min()), 2),
                     image_floor_composition=keys[i],
                     image_median_GPa=round(float(np.median(E)), 2),
                     image_max_GPa=round(float(E.max()), 2)))
    pd.DataFrame(dict(pool=case, composition=keys,
                      pred_modulus_GPa=np.round(E, 2),
                      pred_density_gcc=np.round(R, 3),
                      fitness=np.round(fit, 4))).to_csv(
        os.path.join(DATA, "audit_decoder_image_%s.csv" % case), index=False)

    # are the published candidates and the enumeration winners reachable?
    for f in ("Ca6Zr5Ti2O20", "K12Ca2P10O33", "LaSi3(PO4)5", "Al2(P4O11)3"):
        c = Composition(f)
        reach.append(dict(pool=case, formula=f,
                          in_image=c.reduced_formula in image,
                          in_superset=c.reduced_formula in superset))

cand = pd.read_csv(os.path.join(DATA, "uncertainty_intervals.csv"))
img = {}
for case in ("enamel", "dentin"):
    img[case] = set(pd.read_csv(
        os.path.join(DATA, "audit_decoder_image_%s.csv" % case)).composition)
n_in = 0
for _, r in cand.iterrows():
    pool = "dentin" if r["showcase"] == "dentin" else "enamel"
    c = Composition(r["composition"]).reduced_formula
    ok = c in img[pool]
    n_in += int(ok)
    reach.append(dict(pool=pool, formula=r["composition"], in_image=ok,
                      in_superset=True))
print(f"\npublished candidates inside the decoder's image: {n_in} of {len(cand)}",
      flush=True)
pd.DataFrame(rows).to_csv(os.path.join(DATA, "audit_decoder_image.csv"), index=False)
pd.DataFrame(reach).to_csv(os.path.join(DATA, "audit_decoder_reachability.csv"),
                           index=False)
for r in reach[:8]:
    print(f"  {r['formula']:18s} in image {r['in_image']}  in superset {r['in_superset']}",
          flush=True)
print("\nDECODER_IMAGE_DONE", flush=True)
