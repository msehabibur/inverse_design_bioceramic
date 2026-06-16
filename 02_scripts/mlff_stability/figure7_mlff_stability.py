#!/usr/bin/env python3
"""Figure 7: six-model MLFF convex-hull stability of all 45 GA candidates."""
import os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": ["Arial Narrow", "Arial", "DejaVu Sans"],
    "font.size": 30, "font.weight": "normal",
    "axes.titlesize": 30, "axes.labelsize": 30,
    "xtick.labelsize": 26, "ytick.labelsize": 26, "legend.fontsize": 24,
})
BASE = "/anvil/scratch/x-mrahman2/Purdue_Projects/Inverse_Design_of_Bioceramics_by_Machine_Learning/"
S = BASE + "06_mlff_stability/"
DATA = BASE + "01_data/"
MAN = BASE + "04_manuscript/"
COL = {"enamel": "#1A6090", "dentin": "#D97F33", "implant": "#2E7D5A"}
LAB = {"enamel": "Enamel", "dentin": "Dentin", "implant": "Implant"}
MODELS = ["chgnet", "m3gnet", "mace", "mattersim", "grace", "omni"]
MLAB = {"chgnet": "CHGNet", "m3gnet": "M3GNet", "mace": "MACE",
        "mattersim": "MatterSim", "grace": "GRACE", "omni": "SevenNet"}

df = pd.read_csv(S + "stability_mlff.csv")
lng = pd.read_csv(S + "stability_mlff_long.csv")
# merge predicted modulus by composition
mods = []
for sc, fn in [("enamel", "designed_compositions.csv"), ("dentin", "designed_dentin_compositions.csv"),
               ("implant", "designed_implant_compositions.csv")]:
    t = pd.read_csv(DATA + fn)[["composition", "pred_modulus_GPa"]]; t["showcase"] = sc; mods.append(t)
mods = pd.concat(mods, ignore_index=True)
df = df.merge(mods, on=["composition", "showcase"], how="left")

fig, ax = plt.subplots(2, 2, figsize=(21, 15))

# (a) per-candidate consensus ehull with min-max whiskers, sorted
d = df.sort_values("ehull_median").reset_index(drop=True)
x = np.arange(len(d))
colors = [COL[s] for s in d["showcase"]]
lo = (d["ehull_median"] - d["ehull_min"]).clip(lower=0)
hi = (d["ehull_max"] - d["ehull_median"]).clip(lower=0)
ax[0, 0].bar(x, d["ehull_median"], color=colors, alpha=0.85)
ax[0, 0].errorbar(x, d["ehull_median"], yerr=[lo, hi], fmt="none", ecolor="#333333", lw=1.3, capsize=3)
ax[0, 0].axhline(0.05, ls=":", color="#C0392B", lw=2.5)
ax[0, 0].text(1, 0.075, "Near-stable threshold (0.05)", color="#C0392B", fontsize=22, va="bottom")
ax[0, 0].set_xlabel("Candidate (sorted)")
ax[0, 0].set_ylabel("Energy above hull (eV/atom)")
ax[0, 0].set_title("(a)  Six-model consensus stability", loc="left")
hands = [plt.Rectangle((0, 0), 1, 1, color=COL[s]) for s in COL]
ax[0, 0].legend(hands, [LAB[s] for s in COL], frameon=False, loc="upper left")

# (b) per-model spread (strip)
present = [m for m in MODELS if m in lng["model"].unique()]
for j, m in enumerate(present):
    v = lng[(lng.model == m)]["ehull"].dropna()
    v = v[v.abs() <= 3]
    xj = np.full(len(v), j) + np.random.RandomState(j).uniform(-0.18, 0.18, len(v))
    ax[0, 1].scatter(xj, v, s=45, alpha=0.55, color="#1A6090", edgecolor="white", lw=0.5)
    ax[0, 1].scatter([j], [v.median()], s=260, color="#C0392B", marker="_", lw=4)
ax[0, 1].axhline(0, ls="-", color="#888888", lw=1.5)
ax[0, 1].set_xticks(range(len(present)))
ax[0, 1].set_xticklabels([MLAB[m] for m in present], rotation=30, ha="right")
ax[0, 1].set_ylabel("Energy above hull (eV/atom)")
ax[0, 1].set_title("(b)  Agreement across force fields", loc="left")

# (c) ehull vs predicted modulus
for s in COL:
    sub = df[df.showcase == s]
    ax[1, 0].scatter(sub["pred_modulus_GPa"], sub["ehull_median"], s=140, alpha=0.8,
                     color=COL[s], edgecolor="white", lw=1.0, label=LAB[s])
ax[1, 0].set_xlabel("Predicted Young's modulus (GPa)")
ax[1, 0].set_ylabel("Energy above hull (eV/atom)")
ax[1, 0].set_title("(c)  Stability versus target match", loc="left")
ax[1, 0].legend(frameon=False, loc="best")
ax[1, 0].grid(alpha=0.25)

# (d) distribution by case study
order = ["enamel", "dentin", "implant"]
data = [df[df.showcase == s]["ehull_median"].values for s in order]
parts = ax[1, 1].violinplot(data, showmeans=False, showextrema=False)
for k, b in enumerate(parts["bodies"]):
    b.set_facecolor(COL[order[k]]); b.set_alpha(0.5)
for k, s in enumerate(order):
    yv = df[df.showcase == s]["ehull_median"].values
    ax[1, 1].scatter(np.full(len(yv), k + 1) + np.random.RandomState(k).uniform(-0.08, 0.08, len(yv)),
                     yv, s=80, color=COL[s], edgecolor="white", lw=0.7, zorder=3)
ax[1, 1].set_xticks([1, 2, 3]); ax[1, 1].set_xticklabels([LAB[s] for s in order])
ax[1, 1].set_ylabel("Energy above hull (eV/atom)")
ax[1, 1].set_title("(d)  Distribution by case study", loc="left")

fig.tight_layout(pad=1.6, h_pad=2.6, w_pad=2.2)
out = MAN + "Figure7_mlff_stability.png"
fig.savefig(out, dpi=200)
print("saved", out)
print(f"all above hull: {(df.ehull_median>0.05).sum()}/{len(df)}  mean median={df.ehull_median.mean():.3f}")
