"""Rebuild the enumeration files from the decoder's image.

revision_decoder_image.py already scored every formula the decoder can emit with
the same Scorer the search uses, so the enumeration tables follow from those
files without re-running the scoring. The floors are checked against the values
the image job reported.
"""
import io
import os

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import DATA
img = pd.read_csv(os.path.join(DATA, "audit_decoder_image.csv")).set_index("pool")

frames, floors = [], []
for pool in ("enamel", "dentin"):
    d = pd.read_csv(os.path.join(DATA, "audit_decoder_image_%s.csv" % pool))
    d = d.assign(pool=pool)[["pool", "composition", "pred_modulus_GPa",
                             "pred_density_gcc", "fitness"]]
    frames.append(d)
    i = int(np.argmin(d.pred_modulus_GPa.values))
    floors.append(dict(pool=pool, n_formulas=len(d),
                       modulus_floor_GPa=round(float(d.pred_modulus_GPa.min()), 1),
                       floor_composition=d.composition.iloc[i],
                       modulus_p01_GPa=round(float(np.percentile(d.pred_modulus_GPa, 1)), 1),
                       modulus_median_GPa=round(float(d.pred_modulus_GPa.median()), 1),
                       modulus_max_GPa=round(float(d.pred_modulus_GPa.max()), 1)))
    exp_n = int(img.loc[pool, "n_image"])
    exp_f = float(img.loc[pool, "image_floor_GPa"])
    assert len(d) == exp_n, (pool, len(d), exp_n)
    assert abs(float(d.pred_modulus_GPa.min()) - exp_f) < 0.05, (pool, exp_f)
    print("  %-8s %6d formulas, floor %.2f GPa at %s  (matches the image job)"
          % (pool, len(d), d.pred_modulus_GPa.min(), d.composition.iloc[i]), flush=True)

pd.concat(frames, ignore_index=True).to_csv(
    os.path.join(DATA, "revision_enumeration_full.csv"), index=False)
# revision_enumeration.csv is written by revision_search_baseline.py from the unrounded
# predictions; the image files keep two decimals, so the summary is checked here to
# within the 0.1 GPa rounding of that table and is not rewritten.
ref = pd.read_csv(os.path.join(DATA, "revision_enumeration.csv")).set_index("pool")
for f in floors:
    r = ref.loc[f["pool"]]
    assert int(r.n_formulas) == f["n_formulas"], f["pool"]
    assert r.floor_composition == f["floor_composition"], f["pool"]
    for k in ("modulus_floor_GPa", "modulus_p01_GPa", "modulus_median_GPa", "modulus_max_GPa"):
        assert abs(float(r[k]) - f[k]) <= 0.1 + 1e-9, (f["pool"], k, r[k], f[k])
print("wrote revision_enumeration_full.csv from the image; revision_enumeration.csv agrees",
      flush=True)
print("ENUM_FROM_IMAGE_DONE", flush=True)
