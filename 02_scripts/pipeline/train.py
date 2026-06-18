"""
Step 2 - Train the property models.

Trains three models on the 1721 MP oxides. Each compares random forest vs
gradient boosting by 5-fold cross-validation and keeps the better one:

  model_modulus.joblib     composition + structure -> log10(Young's modulus)
  model_hardness.joblib    composition + structure -> Vickers hardness (Chen)
  model_volperatom.joblib  composition ONLY        -> volume per atom

The vol-per-atom helper lets the inverse-design GA estimate density (and thus
the structural features) for compositions that do not exist yet.

Outputs: model_modulus.joblib, model_hardness.joblib, model_volperatom.joblib
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

df = pd.read_csv(os.path.join(DATA, "mp_data.csv"))
print(f"Loaded {len(df)} materials")

df["composition_obj"] = df["formula"].apply(Composition)

featurizer = make_featurizer()
print("Featurizing compositions...")
df = featurizer.featurize_dataframe(df, col_id="composition_obj", ignore_errors=True)
comp_cols = featurizer.feature_labels()                # composition-only features
full_cols = comp_cols + ["density", "vol_per_atom"]    # + two structural features
df = df.dropna(subset=full_cols)
print(f"  {len(df)} materials featurized")

# Chen-model Vickers hardness from MP's bulk (K) and shear (G) moduli
k = df["G_GPa"] / df["K_GPa"]                          # Pugh's modulus ratio
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
    best_model.fit(X, yy)                              # refit on all data
    joblib.dump({"model": best_model, "features": list(X.columns),
                 "log10_target": log, "name": best_name},
                os.path.join(MODELS, fname))
    print(f"  saved -> {fname}")


# 1. Young's modulus  (composition + structure -> log10 E)
train_one(df[full_cols], df["youngs_modulus_GPa"],
          "Young's modulus", "model_modulus.joblib", "GPa", log=True)

# 2. Vickers hardness (composition + structure -> hardness); Chen model can go
#    negative for very ductile solids -> drop those rows for this model only
hv = df[df["hardness_GPa"] > 0]
train_one(hv[full_cols], hv["hardness_GPa"],
          "Vickers hardness", "model_hardness.joblib", "GPa", log=False)

# 3. Volume-per-atom helper (composition ONLY -> vol/atom)
train_one(df[comp_cols], df["vol_per_atom"],
          "Volume per atom", "model_volperatom.joblib", "A^3", log=False)
