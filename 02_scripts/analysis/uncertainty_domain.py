"""
Uncertainty quantification + applicability domain for the designed candidates.
NO DFT, NO new training data. Each showcase uses its OWN forward model and
training slice (enamel/implant -> model_modulus on the 1,721-oxide pool;
dentin -> model_modulus_dentin on the 531-oxide biocompatible pool).

1. Split-conformal 90% prediction intervals for Young's modulus (the model
   predicts log10 E; we calibrate the absolute-residual quantile on a held-out
   split and report a multiplicative GPa interval).
2. Applicability domain: distance of each designed candidate to its training
   set in standardised composition-feature space.

Outputs: data/uncertainty_intervals.csv, manuscript/Figure8_uncertainty.png
"""
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pymatgen.core import Composition
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # noqa: make paths/features importable
from features import make_featurizer
from paths import DATA, MANUSCRIPT

AMU = 1.66054
COVERAGE = 0.90
featurizer = make_featurizer()
try:
    featurizer.set_n_jobs(8)
except Exception:
    pass


def featurize(df):
    df = df.copy()
    df["composition_obj"] = df["formula"].apply(Composition)
    df = featurizer.featurize_dataframe(df, col_id="composition_obj", ignore_errors=True)
    return df


def calibrate(model_file, vol_file, train_csv):
    """Fit a split-conformal model + AD on a training slice. Returns a bundle."""
    mod = joblib.load(os.path.join(DATA, model_file))
    vol = joblib.load(os.path.join(DATA, vol_file))
    comp_feats, full_feats = vol["features"], mod["features"]
    df = featurize(pd.read_csv(os.path.join(DATA, train_csv)))
    df = df.dropna(subset=full_feats + ["youngs_modulus_GPa"])
    X = df[full_feats].values
    y = np.log10(df["youngs_modulus_GPa"].values)
    Xtr, Xcal, ytr, ycal = train_test_split(X, y, test_size=0.30, random_state=42)
    m = HistGradientBoostingRegressor(random_state=42).fit(Xtr, ytr)
    resid = np.abs(ycal - m.predict(Xcal))
    n = len(resid)
    q = np.sort(resid)[min(int(np.ceil((n + 1) * COVERAGE)), n) - 1]
    cov = float(np.mean(resid <= q))
    mae = float(np.mean(np.abs(10 ** ycal - 10 ** m.predict(Xcal))))
    scaler = StandardScaler().fit(df[comp_feats].values)
    nn = NearestNeighbors(n_neighbors=6).fit(scaler.transform(df[comp_feats].values))
    dtr, _ = nn.kneighbors(scaler.transform(df[comp_feats].values))
    ad_thr = float(np.quantile(dtr[:, 1:].mean(axis=1), 0.99))
    mean_feats = df[comp_feats].mean()
    print(f"  [{model_file}] 90% PI x[{10**-q:.2f},{10**q:.2f}]  cov {cov:.2f}  "
          f"MAE {mae:.1f} GPa  AD_thr {ad_thr:.2f}")
    return dict(m=m, comp_feats=comp_feats, full_feats=full_feats, q=q,
                scaler=scaler, nn=nn, ad_thr=ad_thr, mean_feats=mean_feats, vol=vol)


def score(cands_csv, target, B):
    src = pd.read_csv(os.path.join(DATA, cands_csv))
    comps = [Composition(c) for c in src["composition"]]
    d = featurize(pd.DataFrame({"formula": [c.reduced_formula for c in comps]}))
    cf = B["comp_feats"]
    d[cf] = d[cf].fillna(B["mean_feats"])
    vpa = B["vol"]["model"].predict(d[cf])
    mass = np.array([c.weight / c.num_atoms for c in comps])
    rho = AMU * mass / vpa
    full = d[cf].copy(); full["density"] = rho; full["vol_per_atom"] = vpa
    E = 10 ** B["m"].predict(full[B["full_feats"]].values)
    dd, _ = B["nn"].kneighbors(B["scaler"].transform(d[cf].values))
    ad = dd.mean(axis=1)
    rows = []
    for rank, (c, e, a, fr) in enumerate(zip(src["composition"], E, ad, src["fitness"]), 1):
        lo, hi = e * 10 ** -B["q"], e * 10 ** B["q"]
        rows.append(dict(rank=rank, composition=c, target_E=target,
                         pred_E_GPa=round(float(e), 1), E_lo_GPa=round(float(lo), 1),
                         E_hi_GPa=round(float(hi), 1),
                         target_in_interval=bool(lo <= target <= hi),
                         ad_distance=round(float(a), 2), in_domain=bool(a <= B["ad_thr"]),
                         fitness=fr))
    return rows


print("Calibrating enamel/implant model (mp_data)...")
B_main = calibrate("model_modulus.joblib", "model_volperatom.joblib", "mp_data.csv")
print("Calibrating dentin model (dentin_data)...")
B_dent = calibrate("model_modulus_dentin.joblib", "model_volperatom_dentin.joblib", "dentin_data.csv")

JOBS = [("enamel", "designed_compositions.csv", 85.0, B_main),
        ("dentin", "designed_dentin_compositions.csv", 20.0, B_dent),
        ("implant", "designed_implant_compositions.csv", 110.0, B_main)]

allrows = []
for sc, csv_file, target, B in JOBS:
    for r in score(csv_file, target, B):
        r["showcase"] = sc
        allrows.append(r)
out = pd.DataFrame(allrows)[["showcase", "rank", "composition", "target_E",
    "pred_E_GPa", "E_lo_GPa", "E_hi_GPa", "target_in_interval",
    "ad_distance", "in_domain", "fitness"]]
out.to_csv(os.path.join(DATA, "uncertainty_intervals.csv"), index=False)
for sc, *_ in JOBS:
    s = out[out.showcase == sc]
    print(f"[{sc}] target inside 90% PI {int(s.target_in_interval.sum())}/{len(s)}; "
          f"in-domain {int(s.in_domain.sum())}/{len(s)}")

fig, ax = plt.subplots(1, 3, figsize=(16, 5))
COL = {"enamel": "#1A6090", "dentin": "#D97F33", "implant": "#2E7D5A"}
for j, (sc, _, target, _) in enumerate(JOBS):
    s = out[out.showcase == sc].reset_index(drop=True)
    x = np.arange(len(s)); yv = s["pred_E_GPa"].values
    ax[j].errorbar(x, yv, yerr=[yv - s["E_lo_GPa"].values, s["E_hi_GPa"].values - yv],
                   fmt="o", color=COL[sc], capsize=3, lw=1.2, ms=5)
    ax[j].axhline(target, ls=":", color="#D97F33", lw=1.8, label=f"target {target:.0f} GPa")
    ax[j].set_title(f"{sc.capitalize()}  (90% conformal PI)", loc="left")
    ax[j].set_xlabel("candidate (ranked)"); ax[j].set_ylabel("predicted Young's modulus (GPa)")
    ax[j].legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(MANUSCRIPT, "Figure8_uncertainty.png"), dpi=200, bbox_inches="tight")
print("saved -> Figure8_uncertainty.png + uncertainty_intervals.csv")
