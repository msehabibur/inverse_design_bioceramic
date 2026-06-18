"""
Showcase 2 - Step 2.  Train property models on the dentin-relevant slice.

Same three-model setup as train.py (modulus / hardness / vol-per-atom), but
fit only on the biocompatible-oxide slice from dentin_data.csv (N=531). With a
smaller, chemistry-biased pool than the enamel showcase the CV-R^2 is noisier
(modulus R^2~0.74, hardness R^2~0.47, vol-per-atom R^2~0.60) -- we report the
numbers honestly.

Outputs: model_modulus_dentin.joblib
         model_hardness_dentin.joblib
         model_volperatom_dentin.joblib
"""
import os
import numpy as np
import pandas as pd
import joblib
from pymatgen.core import Composition
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # noqa: make paths/features importable
from features import make_featurizer
from paths import DATA, MODELS

df = pd.read_csv(os.path.join(DATA, "dentin_data.csv"))
print(f"Loaded {len(df)} dentin-relevant materials")

df["composition_obj"] = df["formula"].apply(Composition)
featurizer = make_featurizer()
print("Featurizing...")
df = featurizer.featurize_dataframe(df, col_id="composition_obj",
                                    ignore_errors=True)
comp_cols = featurizer.feature_labels()
full_cols = comp_cols + ["density", "vol_per_atom"]
df = df.dropna(subset=full_cols).reset_index(drop=True)
print(f"  {len(df)} materials after featurization + NaN drop")

# Chen-model Vickers hardness (drop negative-hardness rows for that model)
k = df["G_GPa"] / df["K_GPa"]
df["hardness_GPa"] = 2 * (k ** 2 * df["G_GPa"]) ** 0.585 - 3


def train_one(X, y, name, fname, unit, log=False):
    yy = np.log10(y) if log else y
    cands = {
        "RandomForest": RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1),
        "GradientBoosting": HistGradientBoostingRegressor(random_state=42),
    }
    best_name, best_cv, best_model = None, -1e9, None
    for cn, m in cands.items():
        cv = cross_val_score(m, X, yy, cv=5, scoring="r2", n_jobs=1).mean()
        if cv > best_cv:
            best_name, best_cv, best_model = cn, cv, m
    Xtr, Xte, ytr, yte = train_test_split(X, yy, test_size=0.2, random_state=42)
    best_model.fit(Xtr, ytr)
    pr = best_model.predict(Xte)
    mae = mean_absolute_error(10 ** yte, 10 ** pr) if log else mean_absolute_error(yte, pr)
    print(f"\n[{name}]  best = {best_name}")
    print(f"  CV R^2 = {best_cv:.3f}   held-out R^2 = {r2_score(yte, pr):.3f}"
          f"   MAE = {mae:.2f} {unit}")
    best_model.fit(X, yy)
    joblib.dump({"model": best_model, "features": list(X.columns),
                 "log10_target": log, "name": best_name},
                os.path.join(MODELS, fname))
    print(f"  saved -> {fname}")


# 1. Young's modulus
train_one(df[full_cols], df["youngs_modulus_GPa"],
          "Young's modulus (dentin)", "model_modulus_dentin.joblib", "GPa", log=True)

# 2. Vickers hardness (Chen) -- drop ductile rows
hv = df[df["hardness_GPa"] > 0]
if len(hv) >= 10:
    train_one(hv[full_cols], hv["hardness_GPa"],
              "Vickers hardness (dentin)", "model_hardness_dentin.joblib", "GPa", log=False)
else:
    print(f"\n[Vickers hardness]  only {len(hv)} non-negative rows -- skipped")

# 3. Vol-per-atom helper
train_one(df[comp_cols], df["vol_per_atom"],
          "Volume per atom (dentin)", "model_volperatom_dentin.joblib", "A^3", log=False)
