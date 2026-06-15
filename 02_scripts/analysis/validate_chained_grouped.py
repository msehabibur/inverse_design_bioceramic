"""Reviewer #3/#17: honest validation — chained-pipeline CV + grouped CV."""
import os, sys, numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from features import make_featurizer
from paths import DATA
from pymatgen.core import Composition
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict, cross_val_score
from sklearn.metrics import r2_score, mean_absolute_error

def gbr(): return HistGradientBoostingRegressor(random_state=42)

df = pd.read_csv(os.path.join(DATA, "mp_data.csv"))
df["composition_obj"] = df["formula"].apply(Composition)
fz = make_featurizer()
print("featurizing %d compositions..." % len(df))
df = fz.featurize_dataframe(df, col_id="composition_obj", ignore_errors=True)
comp_cols = fz.feature_labels()
full_cols = comp_cols + ["density", "vol_per_atom"]
df = df.dropna(subset=full_cols + ["youngs_modulus_GPa", "vol_per_atom", "density"]).reset_index(drop=True)
N = len(df); y = np.log10(df["youngs_modulus_GPa"].values)
print("N after dropna =", N)

# [1] standard random 5-fold CV, TRUE structural features
cv_std = cross_val_score(gbr(), df[full_cols], y, cv=5, scoring="r2").mean()
yp_std = cross_val_predict(gbr(), df[full_cols], y, cv=KFold(5, shuffle=True, random_state=42))
r2_std = r2_score(y, yp_std); mae_std = mean_absolute_error(10**y, 10**yp_std)

# [2] CHAINED 5-fold CV: predict vol/atom in-fold, derive density, feed to modulus model
kf = KFold(5, shuffle=True, random_state=42); yt=[]; ypr=[]
for tr, te in kf.split(df):
    vol = gbr().fit(df.iloc[tr][comp_cols], df.iloc[tr]["vol_per_atom"].values)
    vpa = vol.predict(df.iloc[te][comp_cols])
    Xte = df.iloc[te][full_cols].copy()
    Xte["density"] = df.iloc[te]["density"].values * df.iloc[te]["vol_per_atom"].values / vpa
    Xte["vol_per_atom"] = vpa
    mod = gbr().fit(df.iloc[tr][full_cols], y[tr])
    ypr.append(mod.predict(Xte)); yt.append(y[te])
yt=np.concatenate(yt); ypr=np.concatenate(ypr)
r2_chain=r2_score(yt,ypr); mae_chain=mean_absolute_error(10**yt,10**ypr)

# [3]/[4] grouped CV
red = df["composition_obj"].apply(lambda c: c.reduced_formula).values
chs = df["composition_obj"].apply(lambda c: c.chemical_system).values
def grp(groups): return r2_score(y, cross_val_predict(gbr(), df[full_cols], y, cv=GroupKFold(5), groups=groups))
r2_formula = grp(red); r2_chemsys = grp(chs)

print("\n================ VALIDATION RESULTS (Young's modulus) ================")
print("unique reduced formulas: %d  (duplicate rows from polymorphs: %d)" % (len(set(red)), N-len(set(red))))
print("unique chemical systems: %d" % len(set(chs)))
print("[1] standard random 5-fold CV R2 (TRUE density/vol):     %.3f  | MAE %.1f GPa" % (r2_std, mae_std))
print("[2] CHAINED 5-fold CV R2 (PREDICTED density/vol):        %.3f  | MAE %.1f GPa" % (r2_chain, mae_chain))
print("[3] grouped-by-reduced-formula CV R2 (TRUE struct):      %.3f" % r2_formula)
print("[4] grouped-by-chemical-system CV R2 (TRUE struct):      %.3f" % r2_chemsys)
print("=====================================================================")
