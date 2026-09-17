"""Revision analysis R5, for the second referee.

Four things this computes.

1. Where each published coefficient of determination comes from. The 0.74 in the abstract
   and the 0.81 in the Supporting Information both use database structural descriptors.
   They differ in the protocol: 0.74 is the mean of five unshuffled fold scores, 0.81 is a
   single score on pooled out-of-fold predictions with shuffled folds. We reproduce both
   under their own protocol and also report each in linear units, since the modulus is
   regressed in log10.

2. An uncertainty on every metric. Each score is recomputed fold by fold and reported as a
   mean with the standard deviation across folds.

3. Sensitivity of the design to the fitness weights. Every formula the decoder can reach
   has already been scored, so the optimum under any weighting can be found exactly by
   re-ranking that enumeration. We sweep the modulus weight from 0 to 1 and record the
   winner at each step.

4. How many formulas are indistinguishable from the best candidate once the design-time
   error is applied.

Writes 01_data/revision_r2_protocols.csv, revision_metric_uncertainty.csv,
revision_weight_sensitivity.csv and revision_indistinguishable.csv.
"""
import os
import sys
import numpy as np
import pandas as pd
import joblib
from pymatgen.core import Composition
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold, GroupKFold, cross_val_score, cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features import make_featurizer
from paths import DATA

AMU = 1.66054
SEED = 42
NFOLD = 5
TARGETS = {"enamel": (85.0, 3.0), "dentin": (20.0, 2.1), "implant": (110.0, 4.4)}
POOL_OF = {"enamel": "enamel", "implant": "enamel", "dentin": "dentin"}


def gbr():
    return HistGradientBoostingRegressor(random_state=SEED)


def rf():
    return RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=4)


def load(fname):
    df = pd.read_csv(os.path.join(DATA, fname))
    df["composition_obj"] = df["formula"].apply(Composition)
    f = make_featurizer()
    df = f.featurize_dataframe(df, col_id="composition_obj", ignore_errors=True)
    comp = f.feature_labels()
    full = comp + ["density", "vol_per_atom"]
    df = df.dropna(subset=full).reset_index(drop=True)
    k = df["G_GPa"] / df["K_GPa"]
    df["hardness_GPa"] = 2 * (k ** 2 * df["G_GPa"]) ** 0.585 - 3
    df["mean_mass"] = [c.weight / c.num_atoms for c in df["composition_obj"]]
    df["reduced_formula"] = [c.reduced_formula for c in df["composition_obj"]]
    df["chemsys"] = [c.chemical_system for c in df["composition_obj"]]
    return df, comp, full


# ----------------------------------------------------- 1. protocol provenance
rows = []
for pool, fname in (("enamel_implant", "mp_data.csv"), ("dentin", "dentin_data.csv")):
    df, comp, full = load(fname)
    y = np.log10(df["youngs_modulus_GPa"].values)
    X = df[full]

    # the abstract protocol: mean of five unshuffled fold scores, as in train.py
    unshuf = cross_val_score(gbr(), X, y, cv=NFOLD, scoring="r2")
    # the Supporting Information protocol: one score on pooled shuffled out-of-fold
    kf = KFold(NFOLD, shuffle=True, random_state=SEED)
    oof = cross_val_predict(gbr(), X, y, cv=kf)
    shuf_folds = cross_val_score(gbr(), X, y, cv=kf, scoring="r2")

    rows.append(dict(pool=pool, protocol="mean of unshuffled fold scores",
                     descriptors="database", units="log10",
                     r2=round(float(unshuf.mean()), 3),
                     r2_sd=round(float(unshuf.std()), 3),
                     mae_GPa=round(float(mean_absolute_error(
                         10 ** y, 10 ** cross_val_predict(gbr(), X, y, cv=NFOLD))), 2)))
    rows.append(dict(pool=pool, protocol="pooled shuffled out-of-fold",
                     descriptors="database", units="log10",
                     r2=round(float(r2_score(y, oof)), 3),
                     r2_sd=round(float(shuf_folds.std()), 3),
                     mae_GPa=round(float(mean_absolute_error(10 ** y, 10 ** oof)), 2)))
    rows.append(dict(pool=pool, protocol="pooled shuffled out-of-fold",
                     descriptors="database", units="linear GPa",
                     r2=round(float(r2_score(10 ** y, 10 ** oof)), 3),
                     r2_sd=np.nan,
                     mae_GPa=round(float(mean_absolute_error(10 ** y, 10 ** oof)), 2)))
    print(f"{pool}: unshuffled fold mean {unshuf.mean():.3f} "
          f"(sd {unshuf.std():.3f}), pooled shuffled {r2_score(y, oof):.3f}, "
          f"linear-unit pooled {r2_score(10 ** y, 10 ** oof):.3f}", flush=True)

pd.DataFrame(rows).to_csv(os.path.join(DATA, "revision_r2_protocols.csv"), index=False)
print("wrote 01_data/revision_r2_protocols.csv\n", flush=True)

# ----------------------------------------------------- 2. metric uncertainty
def fold_metrics(df, comp, cols, target, kind, mode, est, log):
    y = np.log10(df[target].values) if log else df[target].values
    if kind == "random":
        splits = list(KFold(NFOLD, shuffle=True, random_state=SEED).split(df))
    else:
        g = df["reduced_formula"].values if kind == "formula" else df["chemsys"].values
        splits = list(GroupKFold(NFOLD).split(df, groups=g))
    r2s, maes = [], []
    for tr, te in splits:
        m = est().fit(df.iloc[tr][cols], y[tr])
        Xte = df.iloc[te][cols].copy()
        if mode == "design":
            vol = est().fit(df.iloc[tr][comp], df.iloc[tr]["vol_per_atom"].values)
            vpa = np.clip(vol.predict(df.iloc[te][comp]), 1e-3, None)
            Xte["vol_per_atom"] = vpa
            Xte["density"] = AMU * df.iloc[te]["mean_mass"].values / vpa
        p = m.predict(Xte)
        r2s.append(r2_score(y[te], p))
        maes.append(mean_absolute_error(10 ** y[te], 10 ** p) if log
                    else mean_absolute_error(y[te], p))
    return np.array(r2s), np.array(maes)


urows = []
for pool, fname in (("enamel_implant", "mp_data.csv"), ("dentin", "dentin_data.csv")):
    df, comp, full = load(fname)
    for target, log, est, unit in (("youngs_modulus_GPa", True, gbr, "GPa"),
                                   ("hardness_GPa", False, rf, "GPa"),
                                   ("vol_per_atom", False, gbr, "A^3")):
        sub = df[df["hardness_GPa"] > 0].reset_index(drop=True) \
            if target == "hardness_GPa" else df
        cols = comp if target == "vol_per_atom" else full
        modes = ("true",) if target == "vol_per_atom" else ("true", "design")
        for kind in ("random", "formula", "chemsys"):
            for mode in modes:
                r2s, maes = fold_metrics(sub, comp, cols, target, kind, mode, est, log)
                urows.append(dict(pool=pool, target=target, split=kind, features=mode,
                                  n=len(sub), unit=unit,
                                  r2_mean=round(float(r2s.mean()), 3),
                                  r2_sd=round(float(r2s.std()), 3),
                                  mae_mean=round(float(maes.mean()), 2),
                                  mae_sd=round(float(maes.std()), 2)))
                print(f"  {pool:14s} {target:20s} {kind:8s} {mode:7s} "
                      f"R2 {r2s.mean():.3f}+/-{r2s.std():.3f}  "
                      f"MAE {maes.mean():.2f}+/-{maes.std():.2f} {unit}", flush=True)

pd.DataFrame(urows).to_csv(os.path.join(DATA, "revision_metric_uncertainty.csv"),
                           index=False)
print("wrote 01_data/revision_metric_uncertainty.csv\n", flush=True)

# ----------------------------------------------------- 3. weight sensitivity
enum = pd.read_csv(os.path.join(DATA, "revision_enumeration_full.csv"))
wrows = []
for case, (tE, tR) in TARGETS.items():
    sub = enum[enum.pool == POOL_OF[case]]
    E = sub["pred_modulus_GPa"].values
    R = sub["pred_density_gcc"].values
    names = sub["composition"].values
    for wE in np.round(np.arange(0.0, 1.0001, 0.05), 2):
        f = wE * np.abs(E - tE) / tE + (1 - wE) * np.abs(R - tR) / tR
        i = int(np.argmin(f))
        wrows.append(dict(case=case, w_modulus=wE, w_density=round(1 - wE, 2),
                          best_composition=names[i], best_E_GPa=round(float(E[i]), 1),
                          best_rho_gcc=round(float(R[i]), 3),
                          best_fitness=round(float(f[i]), 4)))
    s = pd.DataFrame([r for r in wrows if r["case"] == case])
    band = s[(s.w_modulus >= 0.3) & (s.w_modulus <= 0.8)]
    print(f"  {case:8s} modulus at the published weight 0.588: "
          f"{float(s[np.isclose(s.w_modulus, 0.60)].best_E_GPa.iloc[0]):.1f} GPa | "
          f"across weights 0.3 to 0.8 the winner spans "
          f"{band.best_E_GPa.min():.1f} to {band.best_E_GPa.max():.1f} GPa, "
          f"{band.best_composition.nunique()} distinct formulas", flush=True)

pd.DataFrame(wrows).to_csv(os.path.join(DATA, "revision_weight_sensitivity.csv"),
                           index=False)
print("wrote 01_data/revision_weight_sensitivity.csv\n", flush=True)

# ----------------------------------------------------- 4. indistinguishability
val = pd.read_csv(os.path.join(DATA, "revision_validation.csv"))
unc = pd.read_csv(os.path.join(DATA, "revision_uncertainty.csv"))
irows = []
for case, (tE, tR) in TARGETS.items():
    pool = "dentin" if case == "dentin" else "enamel_implant"
    v = val[(val.pool == pool) & (val.target == "youngs_modulus_GPa")
            & (val.split == "chemsys") & (val.features == "design")]
    mae = float(v.mae.iloc[0])
    q = float(v.conformal_q.iloc[0])
    best = unc[unc.showcase == case].sort_values("rank").iloc[0]
    Eb = float(best.pred_E_GPa)
    sub = enum[enum.pool == POOL_OF[case]]
    E = sub["pred_modulus_GPa"].values
    within_mae = int(np.sum(np.abs(E - Eb) <= mae))
    lo, hi = Eb * 10 ** -q, Eb * 10 ** q
    within_pi = int(np.sum((E >= lo) & (E <= hi)))
    top15 = unc[unc.showcase == case]
    spread = float(top15.pred_E_GPa.max() - top15.pred_E_GPa.min())
    irows.append(dict(case=case, best_composition=best.composition,
                      best_E_GPa=round(Eb, 1), design_mae_GPa=round(mae, 1),
                      top15_spread_GPa=round(spread, 1),
                      n_enumerated=len(sub),
                      n_within_one_mae=within_mae,
                      pct_within_one_mae=round(100.0 * within_mae / len(sub), 1),
                      n_within_interval=within_pi,
                      pct_within_interval=round(100.0 * within_pi / len(sub), 1)))
    print(f"  {case:8s} best {Eb:.1f} GPa, top-15 spread {spread:.1f} GPa, "
          f"design MAE {mae:.1f} GPa -> {within_mae:,} of {len(sub):,} enumerated "
          f"formulas ({100.0 * within_mae / len(sub):.1f}%) sit within one MAE of the "
          f"best candidate", flush=True)

pd.DataFrame(irows).to_csv(os.path.join(DATA, "revision_indistinguishable.csv"),
                           index=False)
print("wrote 01_data/revision_indistinguishable.csv", flush=True)
print("\nREVISION_REFEREE2_DONE", flush=True)
