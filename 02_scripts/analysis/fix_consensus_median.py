"""Recompute the five-model consensus with the quality exclusion the table declares.

Table S9 calls the negative energies failed calculations, so leaving them in the
median was inconsistent. The consensus is now the median over the models that
returned a positive energy above hull, and the raw five-model median is kept
alongside it as ehull_median_all_models. Two of the 45 candidates move.
"""
import io
import os

import numpy as np
import pandas as pd

ROOT = "/anvil/projects/x-mat260059/Inverse_Design_of_Bioceramics_by_Machine_Learning"
CSV = os.path.join(ROOT, "06_mlff_stability", "stability_mlff.csv")
MODELS = ["chgnet", "m3gnet", "mace", "mattersim", "omni"]

d = pd.read_csv(CSV)
if "ehull_median_all_models" not in d.columns:
    d["ehull_median_all_models"] = d["ehull_median"]
new = d[MODELS].where(d[MODELS] > 0).median(axis=1)
d["n_valid_models"] = d[MODELS].gt(0).sum(axis=1)
moved = (new - d["ehull_median_all_models"]).abs() > 1e-9
d["ehull_median"] = new
d.to_csv(CSV, index=False)

print("consensus median recomputed over the models returning a positive energy")
print("  candidates with a negative model value : %d of %d"
      % (int((d[MODELS].lt(0).any(axis=1)).sum()), len(d)))
print("  candidates with a model that returned no energy : %d"
      % int(d[MODELS].isna().any(axis=1).sum()))
print("  candidates whose median moved          : %d" % int(moved.sum()))
for i in np.where(moved)[0]:
    print("    %-26s %.4f -> %.4f" % (d.composition.iloc[i],
                                      d.ehull_median_all_models.iloc[i],
                                      d.ehull_median.iloc[i]))
print("  mean consensus  %.4f -> %.4f"
      % (d.ehull_median_all_models.mean(), d.ehull_median.mean()))
print("  above 0.05 eV/atom: %d of %d" % (int((d.ehull_median > 0.05).sum()), len(d)))
print("CONSENSUS_FIX_DONE", flush=True)
