"""
Step 3 - Screen REAL zirconia-family compounds.

Queries the Materials Project for real Zr-O compounds, predicts each one's
Young's modulus with the trained model, and shortlists the genuinely stable
ones (using the REAL energy-above-hull from MP). Only compounds the model
never trained on are considered, so the predictions are a fair test.

Output: shortlist.csv
"""
import os
import pandas as pd
import joblib
from mp_api.client import MPRester
from pymatgen.core import Composition
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # noqa: make paths/features importable
from features import make_featurizer
from paths import DATA

KEY = os.environ.get("MP_API_KEY")
if not KEY:
    with open(os.path.join(DATA, "mp_key.txt")) as f:
        KEY = f.read().strip()

# material_ids the model trained on -> exclude them for an honest screen
trained_ids = set(pd.read_csv(os.path.join(DATA, "mp_data.csv"))["material_id"])

print("Querying the Materials Project for zirconia-family (Zr-O) compounds...")
with MPRester(KEY) as mpr:
    docs = mpr.materials.summary.search(
        elements=["Zr", "O"],
        fields=["material_id", "formula_pretty", "energy_above_hull",
                "is_stable", "density", "volume", "nsites"],
    )
print(f"  retrieved {len(docs)} compounds")

rows = []
for d in docs:
    mid = str(d.material_id)
    if mid in trained_ids:                       # skip what the model saw
        continue
    if not d.density or not d.volume or not d.nsites:
        continue
    rows.append({
        "material_id": mid,
        "formula": d.formula_pretty,
        "energy_above_hull": d.energy_above_hull,
        "is_stable": d.is_stable,
        "density": d.density,
        "vol_per_atom": d.volume / d.nsites,
    })
df = pd.DataFrame(rows)
print(f"  {len(df)} compounds not seen during training")

# featurize (same featurizer as train.py) and predict
df["composition_obj"] = df["formula"].apply(Composition)
featurizer = make_featurizer()
print("Featurizing and predicting Young's modulus...")
df = featurizer.featurize_dataframe(df, col_id="composition_obj", ignore_errors=True)

mod = joblib.load(os.path.join(DATA, "model_modulus.joblib"))
feat = mod["features"]
df = df.dropna(subset=feat)
df["pred_modulus_GPa"] = 10 ** mod["model"].predict(df[feat])

# --- screen on REAL stability (energy above hull straight from MP) ---
n_strict = int((df["is_stable"] == True).sum())
keep = df[df["energy_above_hull"] <= 0.05].copy()
print(f"\n{n_strict} strictly stable; {len(keep)} within 0.05 eV/atom of the convex hull")

# keep only the most stable polymorph of each distinct formula
keep = keep.sort_values("energy_above_hull").drop_duplicates("formula", keep="first")
print(f"{len(keep)} distinct compositions after de-duplicating polymorphs")

shortlist = keep.sort_values("pred_modulus_GPa", ascending=False).head(15)
cols = ["material_id", "formula", "energy_above_hull",
        "is_stable", "pred_modulus_GPa"]
shortlist[cols].to_csv(os.path.join(DATA, "shortlist.csv"), index=False)
print("\nStiffest stable zirconia-family compounds (model-predicted modulus):")
print(shortlist[cols].to_string(index=False))
print("\nSaved -> shortlist.csv")
