"""Revision analysis R1 - honest validation of the chained surrogate.

Answers reviewer 1 point 2 and reviewer 3 points 1 and 2:

  * one estimator family per target, three split protocols
      random          KFold(5, shuffle)
      formula         GroupKFold(5) on reduced formula   (no polymorph leakage)
      chemsys         GroupKFold(5) on chemical system   (genuinely new chemistry)
  * two feature modes for the modulus and hardness models
      true            density and vol/atom taken from the Materials Project
      design          vol/atom predicted in-fold from composition only, and
                      density recomputed from the mean atomic mass exactly as
                      the genetic algorithm does at design time
  * out-of-fold split-conformal quantiles taken from the FULL CHAINED model
    under each protocol, with the empirical coverage of those same folds
  * the design-time, chemical-system-grouped interval applied to all 45
    candidates, so the reported interval is the one that matches how the
    model is actually used

Writes 01_data/revision_validation.csv, 01_data/revision_uncertainty.csv,
01_data/revision_novelty.csv and prints a summary block.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from pymatgen.core import Composition
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold, GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features import make_featurizer
from paths import DATA

AMU_PER_A3_TO_GCC = 1.66054          # matches inverse_ga.py
COVERAGE = 0.90
NFOLD = 5
SEED = 42


def gbr():
    return HistGradientBoostingRegressor(random_state=SEED)


def rf():
    return RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=4)


def load(fname):
    df = pd.read_csv(os.path.join(DATA, fname))
    df["composition_obj"] = df["formula"].apply(Composition)
    f = make_featurizer()
    df = f.featurize_dataframe(df, col_id="composition_obj", ignore_errors=True)
    comp_cols = f.feature_labels()
    full_cols = comp_cols + ["density", "vol_per_atom"]
    df = df.dropna(subset=full_cols).reset_index(drop=True)
    k = df["G_GPa"] / df["K_GPa"]
    df["hardness_GPa"] = 2 * (k ** 2 * df["G_GPa"]) ** 0.585 - 3
    df["mean_mass"] = [c.weight / c.num_atoms for c in df["composition_obj"]]
    df["reduced_formula"] = [c.reduced_formula for c in df["composition_obj"]]
    df["chemsys"] = [c.chemical_system for c in df["composition_obj"]]
    return df, comp_cols, full_cols


def splitter(kind, df):
    if kind == "random":
        return KFold(NFOLD, shuffle=True, random_state=SEED).split(df), None
    groups = df["reduced_formula"].values if kind == "formula" else df["chemsys"].values
    return GroupKFold(NFOLD).split(df, groups=groups), groups


def oof_predict(df, comp_cols, full_cols, target, kind, mode, est, log):
    """Out-of-fold predictions under one split protocol and feature mode."""
    y = np.log10(df[target].values) if log else df[target].values
    oof = np.full(len(df), np.nan)
    splits, _ = splitter(kind, df)
    for tr, te in splits:
        Xtr = df.iloc[tr][full_cols]
        m = est().fit(Xtr, y[tr])
        Xte = df.iloc[te][full_cols].copy()
        if mode == "design":
            vol = est().fit(df.iloc[tr][comp_cols], df.iloc[tr]["vol_per_atom"].values)
            vpa = vol.predict(df.iloc[te][comp_cols])
            vpa = np.clip(vpa, 1e-3, None)
            Xte["vol_per_atom"] = vpa
            Xte["density"] = AMU_PER_A3_TO_GCC * df.iloc[te]["mean_mass"].values / vpa
        oof[te] = m.predict(Xte)
    return y, oof


def metrics(y, oof, log):
    ok = ~np.isnan(oof)
    r2 = r2_score(y[ok], oof[ok])
    if log:
        mae = mean_absolute_error(10 ** y[ok], 10 ** oof[ok])
    else:
        mae = mean_absolute_error(y[ok], oof[ok])
    resid = np.abs(y[ok] - oof[ok])
    n = len(resid)
    q = np.sort(resid)[min(int(np.ceil((n + 1) * COVERAGE)), n) - 1]
    cov = float(np.mean(resid <= q))
    return r2, mae, q, cov, n


rows = []
store = {}

for pool, fname in (("enamel_implant", "mp_data.csv"), ("dentin", "dentin_data.csv")):
    print(f"\n#### pool {pool} <- {fname}", flush=True)
    df, comp_cols, full_cols = load(fname)
    print(f"  N = {len(df)}   unique reduced formulas = {df['reduced_formula'].nunique()}"
          f"   unique chemical systems = {df['chemsys'].nunique()}", flush=True)

    # self-check: the density identity used at design time must hold on real data
    rho_id = AMU_PER_A3_TO_GCC * df["mean_mass"].values / df["vol_per_atom"].values
    err = np.abs(rho_id - df["density"].values) / df["density"].values
    print(f"  density identity check: max relative error {err.max():.2e}"
          f"  median {np.median(err):.2e}", flush=True)

    for target, log, est, unit in (("youngs_modulus_GPa", True, gbr, "GPa"),
                                   ("hardness_GPa", False, rf, "GPa"),
                                   ("vol_per_atom", False, gbr, "A^3")):
        sub = df[df["hardness_GPa"] > 0].reset_index(drop=True) if target == "hardness_GPa" else df
        modes = ("true",) if target == "vol_per_atom" else ("true", "design")
        cols = comp_cols if target == "vol_per_atom" else full_cols
        for kind in ("random", "formula", "chemsys"):
            for mode in modes:
                y, oof = oof_predict(sub, comp_cols, cols, target, kind, mode, est, log)
                r2, mae, q, cov, n = metrics(y, oof, log)
                rows.append(dict(pool=pool, target=target, split=kind, features=mode,
                                 n=n, r2=round(r2, 3), mae=round(mae, 2), unit=unit,
                                 conformal_q=round(float(q), 4),
                                 pi_lo_mult=round(float(10 ** -q), 3) if log else np.nan,
                                 pi_hi_mult=round(float(10 ** q), 3) if log else np.nan,
                                 coverage=round(cov, 3)))
                print(f"  {target:22s} {kind:8s} {mode:7s} "
                      f"R2 {r2:6.3f}  MAE {mae:7.2f} {unit}  q {q:.4f}  cov {cov:.3f}",
                      flush=True)
                if target == "youngs_modulus_GPa":
                    store[(pool, kind, mode)] = float(q)

    store[(pool, "df")] = df
    store[(pool, "comp_cols")] = comp_cols
    store[(pool, "full_cols")] = full_cols

val = pd.DataFrame(rows)
val.to_csv(os.path.join(DATA, "revision_validation.csv"), index=False)
print("\nwrote 01_data/revision_validation.csv", flush=True)

# ---------------------------------------------------------------- candidates
# Apply the design-time, chemical-system-grouped quantile: the honest interval.
cand = pd.read_csv(os.path.join(DATA, "uncertainty_intervals.csv"))
frozen = pd.read_csv(os.path.join(DATA, "designed_dentin_compositions.csv"))
assert set(frozen["composition"]) == set(cand[cand.showcase == "dentin"]["composition"]), \
    "dentin candidate list in uncertainty_intervals.csv is not the frozen GA list"

out = []
for _, r in cand.iterrows():
    pool = "dentin" if r["showcase"] == "dentin" else "enamel_implant"
    q = store[(pool, "chemsys", "design")]
    e = float(r["pred_E_GPa"])
    lo, hi = e * 10 ** -q, e * 10 ** q
    out.append(dict(showcase=r["showcase"], rank=int(r["rank"]),
                    composition=r["composition"], target_E=float(r["target_E"]),
                    pred_E_GPa=e, E_lo_GPa=round(lo, 1), E_hi_GPa=round(hi, 1),
                    target_in_interval=bool(lo <= float(r["target_E"]) <= hi),
                    conformal_q=round(q, 4), split="chemsys", features="design"))
unc = pd.DataFrame(out)
unc.to_csv(os.path.join(DATA, "revision_uncertainty.csv"), index=False)
print("wrote 01_data/revision_uncertainty.csv", flush=True)
for sc in ("enamel", "dentin", "implant"):
    s = unc[unc.showcase == sc]
    print(f"  {sc:8s} target in interval: {int(s.target_in_interval.sum())}/{len(s)}"
          f"   interval x[{10 ** -s.conformal_q.iloc[0]:.2f}, {10 ** s.conformal_q.iloc[0]:.2f}]",
          flush=True)

# ------------------------------------------------------------------ novelty
# Reviewer 1 point 6: overlap with training data by reduced formula and chemsys.
nov = []
for _, r in cand.iterrows():
    pool = "dentin" if r["showcase"] == "dentin" else "enamel_implant"
    df = store[(pool, "df")]
    c = Composition(r["composition"])
    nov.append(dict(showcase=r["showcase"], rank=int(r["rank"]),
                    composition=r["composition"],
                    reduced_formula=c.reduced_formula,
                    chemsys=c.chemical_system,
                    formula_in_training=bool(c.reduced_formula in set(df["reduced_formula"])),
                    chemsys_in_training=bool(c.chemical_system in set(df["chemsys"])),
                    n_training_same_chemsys=int((df["chemsys"] == c.chemical_system).sum())))
nv = pd.DataFrame(nov)
nv.to_csv(os.path.join(DATA, "revision_novelty.csv"), index=False)
print("wrote 01_data/revision_novelty.csv", flush=True)
print(f"  reduced-formula overlap with training set: {int(nv.formula_in_training.sum())}/45")
print(f"  chemical-system overlap with training set: {int(nv.chemsys_in_training.sum())}/45",
      flush=True)

# ------------------------------------------- applicability domain, fixed k
# The shipped script compared a 5-neighbour training threshold against a
# 6-neighbour candidate distance. Use 5 excluding self on both sides.
ad_rows = []
for pool in ("enamel_implant", "dentin"):
    df = store[(pool, "df")]
    comp_cols = store[(pool, "comp_cols")]
    sc = StandardScaler().fit(df[comp_cols].values)
    Z = sc.transform(df[comp_cols].values)
    nn = NearestNeighbors(n_neighbors=6).fit(Z)
    dtr, _ = nn.kneighbors(Z)
    thr = float(np.quantile(dtr[:, 1:6].mean(axis=1), 0.99))
    sel = cand[cand.showcase == "dentin"] if pool == "dentin" else \
        cand[cand.showcase != "dentin"]
    f = make_featurizer()
    d = pd.DataFrame({"composition_obj": [Composition(x) for x in sel["composition"]]})
    d = f.featurize_dataframe(d, col_id="composition_obj", ignore_errors=True)
    dd, _ = nn.kneighbors(sc.transform(d[comp_cols].values))
    dist = dd[:, :5].mean(axis=1)
    for (_, r), a in zip(sel.iterrows(), dist):
        ad_rows.append(dict(showcase=r["showcase"], rank=int(r["rank"]),
                            composition=r["composition"],
                            ad_distance=round(float(a), 2),
                            ad_threshold=round(thr, 2), in_domain=bool(a <= thr)))
ad = pd.DataFrame(ad_rows)
ad.to_csv(os.path.join(DATA, "revision_domain.csv"), index=False)
print("wrote 01_data/revision_domain.csv", flush=True)
print(f"  in domain (k=5 both sides): {int(ad.in_domain.sum())}/45", flush=True)
print("\nREVISION_VALIDATION_DONE", flush=True)
