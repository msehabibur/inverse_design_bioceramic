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

DATA = "/anvil/projects/x-mat260059/Inverse_Design_of_Bioceramics_by_Machine_Learning/01_data"
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
pd.DataFrame(floors).to_csv(os.path.join(DATA, "revision_enumeration.csv"), index=False)
print("wrote revision_enumeration_full.csv and revision_enumeration.csv from the image",
      flush=True)
print("ENUM_FROM_IMAGE_DONE", flush=True)
