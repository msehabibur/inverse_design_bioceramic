"""
Figure 2 of the new minimal paper: 6-panel summary of the ENAMEL showcase
arranged as 2 rows x 3 columns (parallel to figure3_dentin_panels.py).

  Row 1 (forward model):
    (a)  modulus parity (5-fold CV)
    (b)  hardness parity (5-fold CV, Chen-model Vickers)
    (c)  density parity  (CV via V/atom model, then derived)
  Row 2 (inverse design):
    (d)  GA convergence  (best fitness per generation)
    (e)  GA population evolution  (E vs rho, coloured by generation, with enamel star)
    (f)  top-15 candidates  (predicted modulus bars vs enamel target)

Output: figure2_enamel_panels.png  (single PNG, dpi=220)

Same styling decisions as figure3_dentin_panels.py:
  - Arial Narrow, 22 pt, no bold, no suptitle, no R^2 text boxes
  - y=x line in parity panels without legend entry, equal aspect
  - target line gets a white-on-orange label only in panel (f)
"""
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family":       ["Arial Narrow", "Arial", "Helvetica Neue", "DejaVu Sans"],
    "font.size":         32,
    "font.weight":       "normal",
    "axes.titlesize":    32,
    "axes.titleweight":  "normal",
    "axes.labelsize":    32,
    "axes.labelweight":  "normal",
    "xtick.labelsize":   30,
    "ytick.labelsize":   30,
    "legend.fontsize":   30,
    "figure.titlesize":  32,
    "figure.titleweight":"normal",
})

from sklearn.base import clone
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error
from pymatgen.core import Composition
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # noqa: make paths/features importable
from features import make_featurizer
from paths import DATA, MANUSCRIPT
from pymoo.core.problem import Problem
from pymoo.algorithms.soo.nonconvex.ga import GA
from pymoo.optimize import minimize

BLUE   = "#1A6090"
ORANGE = "#D97F33"
GREY   = "#777777"
AMU_PER_A3_TO_GCC = 1.66054

# Enamel targets + GA cation pool (matches inverse_ga.py)
TARGET_E, TARGET_H, TARGET_RHO = 85.0, 4.0, 3.0
W_E, W_H, W_RHO = 0.588, 0.0, 0.412   # hardness excluded (matches production GA)
CHARGE = {"Zr": 4, "Ti": 4, "Si": 4, "Ce": 4, "P": 5,
          "Al": 3, "Y": 3, "La": 3, "Ca": 2, "Mg": 2, "Sr": 2}
CATIONS = list(CHARGE)


# --------------------------------------------------------------------------
# Load enamel training data + featurize once
# --------------------------------------------------------------------------
print("Loading mp_data.csv and featurizing (one-time) ...")
df = pd.read_csv(os.path.join(DATA, "mp_data.csv"))
df["composition_obj"] = df["formula"].apply(Composition)
featurizer = make_featurizer()
df = featurizer.featurize_dataframe(df, col_id="composition_obj", ignore_errors=True)
comp_cols = featurizer.feature_labels()
full_cols = comp_cols + ["density", "vol_per_atom"]
df = df.dropna(subset=full_cols).reset_index(drop=True)

k = df["G_GPa"] / df["K_GPa"]
df["hardness_GPa"] = 2 * (k ** 2 * df["G_GPa"]) ** 0.585 - 3

mod  = joblib.load(os.path.join(DATA, "model_modulus.joblib"))
hard = joblib.load(os.path.join(DATA, "model_hardness.joblib"))
vol  = joblib.load(os.path.join(DATA, "model_volperatom.joblib"))


# --------------------------------------------------------------------------
# CV predictions for the three parity panels
# --------------------------------------------------------------------------
print("CV predictions for modulus / hardness / V-per-atom ...")
X_E = df[mod["features"]]
y_E_log = np.log10(df["youngs_modulus_GPa"].values)
pred_E_log = cross_val_predict(clone(mod["model"]), X_E, y_E_log, cv=5, n_jobs=1)
actual_E, pred_E = 10 ** y_E_log, 10 ** pred_E_log
r2_E = r2_score(y_E_log, pred_E_log)

hv = df[df["hardness_GPa"] > 0].reset_index(drop=True)
X_H = hv[hard["features"]]; y_H = hv["hardness_GPa"].values
pred_H_cv = cross_val_predict(clone(hard["model"]), X_H, y_H, cv=5, n_jobs=1)
r2_H = r2_score(y_H, pred_H_cv)
mae_H = mean_absolute_error(y_H, pred_H_cv)

X_V = df[vol["features"]]; y_V = df["vol_per_atom"].values
pred_V_cv = cross_val_predict(clone(vol["model"]), X_V, y_V, cv=5, n_jobs=1)
mean_mass = np.array([Composition(f).weight / Composition(f).num_atoms
                      for f in df["formula"]])
pred_rho = AMU_PER_A3_TO_GCC * mean_mass / pred_V_cv
actual_rho = df["density"].values
r2_rho = r2_score(actual_rho, pred_rho)
mae_rho = mean_absolute_error(actual_rho, pred_rho)


# --------------------------------------------------------------------------
# Re-run enamel GA for population + convergence (same setup as inverse_ga.py)
# --------------------------------------------------------------------------
def decode(x):
    amt = dict(zip(CATIONS, x))
    top = sorted(amt.items(), key=lambda kv: -kv[1])[:4]
    keep = {e: a for e, a in top if a > 0.10}
    if not keep:
        keep = {"Zr": 1.0}
    o = sum(a * CHARGE[e] for e, a in keep.items()) / 2.0
    return Composition({**keep, "O": o})


def score_batch(comps):
    d = pd.DataFrame({"composition_obj": comps})
    d = featurizer.featurize_dataframe(d, col_id="composition_obj",
                                       ignore_errors=True)
    cc = vol["features"]
    d[cc] = d[cc].fillna(d[cc].mean())
    vpa = vol["model"].predict(d[cc])
    mm  = np.array([c.weight / c.num_atoms for c in comps])
    rho = AMU_PER_A3_TO_GCC * mm / vpa
    full = d[cc].copy()
    full["density"] = rho
    full["vol_per_atom"] = vpa
    E = 10 ** mod["model"].predict(full[mod["features"]])
    H = hard["model"].predict(full[mod["features"]])
    fit = (W_E * np.abs(E - TARGET_E) / TARGET_E
           + W_H * np.abs(H - TARGET_H) / TARGET_H
           + W_RHO * np.abs(rho - TARGET_RHO) / TARGET_RHO)
    return fit, E, H, rho


class EnamelDesign(Problem):
    def __init__(self):
        super().__init__(n_var=len(CATIONS), n_obj=1, xl=0.0, xu=1.0)
    def _evaluate(self, X, out, *args, **kwargs):
        fit, *_ = score_batch([decode(row) for row in X])
        out["F"] = fit.reshape(-1, 1)


print("Re-running GA (seed=42, 25 gens, pop=60) ...")
res = minimize(EnamelDesign(),
               GA(pop_size=60, eliminate_duplicates=True),
               ("n_gen", 25), seed=42, save_history=True, verbose=False)

gen_best_fit = []
all_gen, all_E, all_rho = [], [], []
for g, algo in enumerate(res.history):
    comps = [decode(r) for r in algo.pop.get("X")]
    fit, E, H, rho = score_batch(comps)
    gen_best_fit.append(fit.min())
    all_gen.extend([g] * len(E))
    all_E.extend(E.tolist())
    all_rho.extend(rho.tolist())
all_gen = np.array(all_gen); all_E = np.array(all_E); all_rho = np.array(all_rho)

ds = pd.read_csv(os.path.join(DATA, "designed_compositions.csv"))
ds["short"] = [str(i + 1) for i in range(len(ds))]


# --------------------------------------------------------------------------
# Build the 2x3 figure
# --------------------------------------------------------------------------
print("Composing 2x3 figure ...")
fig, axes = plt.subplots(2, 3, figsize=(20, 13.5))


# (a) modulus ---------------------------------------------------------------
ax = axes[0, 0]
hi = max(actual_E.max(), pred_E.max()) * 1.05
lo = -0.04 * hi
ax.plot([lo, hi], [lo, hi], "--", color="#555555", lw=1.2)
ax.scatter(actual_E, pred_E, s=20, alpha=0.6, color=BLUE, edgecolor="none")
ax.axvline(TARGET_E, ls=":", color=ORANGE, lw=1.4, label=f"target = {TARGET_E:.0f} GPa")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_aspect("equal", "box")
ax.set_yticks(ax.get_xticks())
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("DFT Young's modulus (GPa)")
ax.set_ylabel("ML-predicted modulus (GPa)")
ax.set_title("(a)  Modulus", loc="left")
ax.legend(loc="lower right", frameon=False)


# (b) hardness --------------------------------------------------------------
ax = axes[0, 1]
H_HI = 25
H_LO = -0.04 * H_HI  # slight negative buffer; no points cropped
ax.plot([H_LO, H_HI], [H_LO, H_HI], "--", color="#555555", lw=1.2)
ax.scatter(y_H, pred_H_cv, s=20, alpha=0.6, color=BLUE, edgecolor="none")
ax.axvline(TARGET_H, ls=":", color=ORANGE, lw=1.4, label=f"target = {TARGET_H:.1f} GPa")
ax.set_xlim(H_LO, H_HI); ax.set_ylim(H_LO, H_HI)
ax.set_aspect("equal", "box")
ax.set_yticks(ax.get_xticks())
ax.set_xlim(H_LO, H_HI); ax.set_ylim(H_LO, H_HI)
ax.set_xlabel("Chen-model Vickers H (GPa)")
ax.set_ylabel("ML-predicted hardness (GPa)")
ax.set_title("(b)  Hardness", loc="left")
ax.legend(loc="lower right", frameon=False)


# (c) density ---------------------------------------------------------------
ax = axes[0, 2]
hi = max(actual_rho.max(), pred_rho.max()) * 1.05
lo = -0.04 * hi
ax.plot([lo, hi], [lo, hi], "--", color="#555555", lw=1.2)
ax.scatter(actual_rho, pred_rho, s=20, alpha=0.6, color=BLUE, edgecolor="none")
ax.axvline(TARGET_RHO, ls=":", color=ORANGE, lw=1.4, label=f"target = {TARGET_RHO:.1f} g/cm³")
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_aspect("equal", "box")
ax.set_yticks(ax.get_xticks())
ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
ax.set_xlabel("Materials Project density (g/cm³)")
ax.set_ylabel("Derived density from V/atom model (g/cm³)")
ax.set_title("(c)  Density", loc="left")
ax.legend(loc="lower right", frameon=False)


# (d) GA convergence ---------------------------------------------------------
ax = axes[1, 0]
ax.plot(range(len(gen_best_fit)), gen_best_fit, "-o", color=BLUE, lw=1.8, ms=6)
ax.set_xlabel("Generation")
ax.set_ylabel("Best fitness  (lower = closer to enamel)")
ax.set_title("(d)  Genetic algorithm convergence", loc="left")
ax.grid(alpha=0.3)


# (e) GA population evolution -----------------------------------------------
ax = axes[1, 1]
sc = ax.scatter(all_E, all_rho, c=all_gen, s=14, alpha=0.6,
                cmap="viridis", edgecolor="none")
ax.axvline(TARGET_E,   ls="--", color=ORANGE, lw=1.3)
ax.axhline(TARGET_RHO, ls="--", color=ORANGE, lw=1.3)
ax.scatter([TARGET_E], [TARGET_RHO], marker="*", s=260, color=ORANGE,
           edgecolor="k", lw=0.8, zorder=5, label="enamel target")
cb = plt.colorbar(sc, ax=ax, shrink=0.85); cb.set_label("Generation")
ax.set_xlabel("Predicted modulus (GPa)")
ax.set_ylabel("Predicted density (g/cm³)")
ax.set_title("(e)  Genetic algorithm population evolution", loc="left")
ax.legend(loc="upper right", frameon=False)


# (f) top-15 candidates: predicted modulus vs target ------------------------
ax = axes[1, 2]
ax.bar(ds["short"], ds["pred_modulus_GPa"], color=BLUE, alpha=0.85)
ax.axhline(TARGET_E, ls="--", color=ORANGE, lw=2.0)
ax.text(0.98, TARGET_E, f"target = {TARGET_E:.0f} GPa",
        ha="right", va="center", color="white",
        bbox=dict(boxstyle="round,pad=0.25", fc=ORANGE, ec="none"),
        transform=ax.get_yaxis_transform())
ax.set_xlabel("Candidate (ranked)")
ax.set_ylabel("Predicted modulus (GPa)")
ax.set_title("(f)  Top-15 candidates: modulus", loc="left")
ax.tick_params(axis="x", rotation=0, labelsize=28)


fig.tight_layout()
out = os.path.join(MANUSCRIPT, "Figure4_enamel_panels.png")
fig.savefig(out, dpi=220)
print(f"\nsaved -> {out}")
