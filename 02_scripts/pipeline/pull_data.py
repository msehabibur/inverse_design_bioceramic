"""
Step 1 - Pull training data from the Materials Project.

Downloads every oxygen-containing material that has computed elastic
properties, plus its stability (energy above hull). This broad set of
oxides is what the model learns the "composition -> stiffness" pattern from.

Output: mp_data.csv
"""
import os
import pandas as pd
from mp_api.client import MPRester
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # noqa: make paths/features importable
from paths import DATA

# --- API key: from environment variable, or the local mp_key.txt file ---
KEY = os.environ.get("MP_API_KEY")
if not KEY:
    with open(os.path.join(DATA, "mp_key.txt")) as f:
        KEY = f.read().strip()

print("Querying the Materials Project for oxides with elastic data...")
with MPRester(KEY) as mpr:
    docs = mpr.materials.summary.search(
        elements=["O"],                 # must contain oxygen
        has_props=["elasticity"],       # must have computed elastic data
        fields=[
            "material_id", "formula_pretty",
            "energy_above_hull", "is_stable",
            "bulk_modulus", "shear_modulus", "nelements",
            "density", "volume", "nsites",
        ],
    )
print(f"  retrieved {len(docs)} materials")

rows = []
for d in docs:
    # bulk_modulus / shear_modulus are dicts: {"voigt", "reuss", "vrh"}
    K = d.bulk_modulus.get("vrh") if d.bulk_modulus else None
    G = d.shear_modulus.get("vrh") if d.shear_modulus else None
    if K is None or G is None or K <= 0 or G <= 0:
        continue
    # Young's modulus from bulk (K) and shear (G):  E = 9KG / (3K + G)
    E = 9 * K * G / (3 * K + G)
    if not d.density or not d.volume or not d.nsites:
        continue
    rows.append({
        "material_id": str(d.material_id),
        "formula": d.formula_pretty,
        "K_GPa": K,
        "G_GPa": G,
        "youngs_modulus_GPa": E,
        "energy_above_hull": d.energy_above_hull,
        "is_stable": d.is_stable,
        "nelements": d.nelements,
        "density": d.density,                  # g/cm^3  (structural)
        "vol_per_atom": d.volume / d.nsites,    # A^3/atom (structural)
    })

df = pd.DataFrame(rows)
out = os.path.join(DATA, "mp_data.csv")
df.to_csv(out, index=False)
print(f"Saved {len(df)} clean rows -> {out}")
print(df[["formula", "youngs_modulus_GPa", "energy_above_hull"]].head())
