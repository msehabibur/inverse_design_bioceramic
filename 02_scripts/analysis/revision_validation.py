"""Revision analysis R1 - validation of the chained surrogate.

Answers reviewer 1 point 2 and reviewer 3 points 1 and 2:

  * the released model chain of each pool (chain.py: estimator class and settings
    read from 01_data/model_*.joblib), refitted inside every fold on the rows
    train.py uses, under three split protocols
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

Run under 05_config_envs/site_venv (scikit-learn 1.6.1, the version the models were
pickled with).

Writes 01_data/revision_validation.csv, revision_metric_uncertainty.csv,
01_data/revision_uncertainty.csv,
01_data/revision_novelty.csv and prints a summary block.
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from pymatgen.core import Composition
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features import make_featurizer
from paths import DATA
from chain import (AMU_PER_A3_TO_GCC, POOLS, Chain, check_released, group_folds,
                   load_pool, released)

COVERAGE = 0.90
NFOLD = 5
SEED = 42


def splitter(kind, df, pool):
    if kind == "random":
        return list(KFold(NFOLD, shuffle=True, random_state=SEED).split(df))
    groups = df["reduced_formula"].values if kind == "formula" else df["chemsys"].values
    return group_folds(groups, "%s/%s" % (pool, kind), NFOLD)


def oof_chain(df, comp_cols, full_cols, recs, splits):
    """Out-of-fold predictions of the released chain, refitted inside every fold.

    Folds are drawn on the whole pool. In each fold the volume-per-atom helper and the
    modulus model are fitted on the training part and the hardness model on its rows
    with a positive Chen hardness, as train.py does; the held-out rows are predicted
    with database and with predicted structural inputs.
    """
    out = {k: np.full(len(df), np.nan) for k in
           ("E_true", "E_design", "H_true", "H_design", "V")}
    fold = np.full(len(df), -1)
    for i, (tr, te) in enumerate(splits):
        c = Chain(recs, df, comp_cols, full_cols, tr)
        t = df.iloc[te]
        out["E_true"][te], out["H_true"][te], out["V"][te] = c.predict(t, "true")
        out["E_design"][te], out["H_design"][te], _ = c.predict(t, "design")
        fold[te] = i
    return out, fold


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


def fold_scores(y, oof, fold, log):
    r2s, maes = [], []
    for i in range(NFOLD):
        s = fold == i
        r2s.append(r2_score(y[s], oof[s]))
        maes.append(mean_absolute_error(10 ** y[s], 10 ** oof[s]) if log
                    else mean_absolute_error(y[s], oof[s]))
    return np.array(r2s), np.array(maes)


rows, urows, parity_rows = [], [], []
store = {}

for pool, fname, sfx in POOLS:
    print(f"\n#### pool {pool} <- {fname}", flush=True)
    df, comp_cols, full_cols = load_pool(fname)
    recs = released(sfx)
    print(f"  N = {len(df)}   unique reduced formulas = {df['reduced_formula'].nunique()}"
          f"   unique chemical systems = {df['chemsys'].nunique()}", flush=True)
    print("  released chain: " + ", ".join(f"{k} {type(r['model']).__name__}"
                                           for k, r in recs.items()), flush=True)
    worst = check_released(df, comp_cols, full_cols, recs)
    print("  refit on the whole pool reproduces the released models, max |difference| "
          + ", ".join(f"{k} {v:.2e}" for k, v in worst.items()), flush=True)

    # self-check: the density identity used at design time must hold on real data
    rho_id = AMU_PER_A3_TO_GCC * df["mean_mass"].values / df["vol_per_atom"].values
    err = np.abs(rho_id - df["density"].values) / df["density"].values
    print(f"  density identity check: max relative error {err.max():.2e}"
          f"  median {np.median(err):.2e}", flush=True)

    hpos = df["hardness_GPa"].values > 0
    for kind in ("random", "formula", "chemsys"):
        oof, fold = oof_chain(df, comp_cols, full_cols, recs, splitter(kind, df, pool))
        if kind == "random":
            # the parity panels of the case-study figures plot these
            vpa = np.clip(oof["V"], 1e-3, None)
            parity_rows.append(pd.DataFrame(dict(
                pool=pool, material_id=df["material_id"], formula=df["formula"],
                E_GPa=df["youngs_modulus_GPa"], E_pred_GPa=10 ** oof["E_true"],
                H_GPa=df["hardness_GPa"].where(hpos),
                H_pred_GPa=pd.Series(oof["H_true"]).where(hpos),
                density_gcc=df["density"],
                density_pred_gcc=AMU_PER_A3_TO_GCC * df["mean_mass"].values / vpa)))
        cases = [("youngs_modulus_GPa", True, "GPa", "true", np.log10(df["youngs_modulus_GPa"].values), oof["E_true"], slice(None)),
                 ("youngs_modulus_GPa", True, "GPa", "design", np.log10(df["youngs_modulus_GPa"].values), oof["E_design"], slice(None)),
                 ("hardness_GPa", False, "GPa", "true", df["hardness_GPa"].values, oof["H_true"], hpos),
                 ("hardness_GPa", False, "GPa", "design", df["hardness_GPa"].values, oof["H_design"], hpos),
                 ("vol_per_atom", False, "A^3", "true", df["vol_per_atom"].values, oof["V"], slice(None))]
        for target, log, unit, mode, y, p, sel in cases:
            y, p, f_ = y[sel], p[sel], fold[sel]
            r2, mae, q, cov, n = metrics(y, p, log)
            r2s, maes = fold_scores(y, p, f_, log)
            rows.append(dict(pool=pool, target=target, split=kind, features=mode,
                             n=n, r2=round(r2, 3), mae=round(mae, 2), unit=unit,
                             conformal_q=round(float(q), 4),
                             pi_lo_mult=round(float(10 ** -q), 3) if log else np.nan,
                             pi_hi_mult=round(float(10 ** q), 3) if log else np.nan,
                             coverage=round(cov, 3)))
            urows.append(dict(pool=pool, target=target, split=kind, features=mode,
                              n=n, unit=unit,
                              r2_mean=round(float(r2s.mean()), 3),
                              r2_sd=round(float(r2s.std()), 3),
                              mae_mean=round(float(maes.mean()), 2),
                              mae_sd=round(float(maes.std()), 2)))
            print(f"  {target:22s} {kind:8s} {mode:7s} "
                  f"R2 {r2:6.3f} (folds {r2s.mean():.3f}+/-{r2s.std():.3f})  "
                  f"MAE {mae:7.2f} {unit}  q {q:.4f}", flush=True)
            if target == "youngs_modulus_GPa":
                store[(pool, kind, mode)] = float(q)

    store[(pool, "df")] = df
    store[(pool, "comp_cols")] = comp_cols
    store[(pool, "full_cols")] = full_cols

pd.DataFrame(urows).to_csv(os.path.join(DATA, "revision_metric_uncertainty.csv"),
                           index=False)
print("\nwrote 01_data/revision_metric_uncertainty.csv", flush=True)
pd.concat(parity_rows).to_csv(os.path.join(DATA, "revision_parity_oof.csv"), index=False)
print("wrote 01_data/revision_parity_oof.csv", flush=True)
val = pd.DataFrame(rows)
val.to_csv(os.path.join(DATA, "revision_validation.csv"), index=False)
print("\nwrote 01_data/revision_validation.csv", flush=True)

# ---------------------------------------------------------------- candidates
# Apply the design-time, chemical-system-grouped quantile.
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
