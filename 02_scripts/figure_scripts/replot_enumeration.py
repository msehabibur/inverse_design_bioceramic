"""Restyle Figure9_enumeration.png from the saved CSV (house style, no recompute)."""
import os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # noqa: make paths/features importable
from paths import DATA, MANUSCRIPT

plt.rcParams.update({
    "font.family": ["Arial Narrow", "Arial", "Helvetica Neue", "DejaVu Sans"],
    "font.size": 28, "font.weight": "normal",
    "axes.titlesize": 28, "axes.titleweight": "normal",
    "axes.labelsize": 28, "axes.labelweight": "normal",
    "xtick.labelsize": 26, "ytick.labelsize": 26, "legend.fontsize": 26,
})
COL = {"enamel": "#1A6090", "dentin": "#D97F33", "implant": "#2E7D5A"}
TARGET = {"enamel": 85.0, "dentin": 20.0, "implant": 110.0}

df = pd.read_csv(os.path.join(DATA, "enumeration_modulus.csv"))
fig, ax = plt.subplots(1, 3, figsize=(19, 6.8))
for j, sc in enumerate(["enamel", "dentin", "implant"]):
    v = df[df.showcase == sc]["pred_modulus_GPa"].values
    floor = float(v.min())
    ax[j].hist(v, bins=60, color=COL[sc], alpha=0.85)
    ax[j].axvline(TARGET[sc], ls=":", color="black", lw=2.2, label=f"Target {TARGET[sc]:.0f} GPa")
    ax[j].axvline(floor, ls="--", color="#C0392B", lw=2, label=f"Floor {floor:.0f} GPa")
    ax[j].set_title(f"{sc.capitalize()}: {len(v):,} Reachable Oxides", loc="left")
    ax[j].set_xlabel("Predicted Young's Modulus (GPa)")
    ax[j].set_ylabel("Count")
    ax[j].legend(frameon=False, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(MANUSCRIPT, "Figure9_enumeration.png"), dpi=200, bbox_inches="tight")
print("saved restyled Figure9_enumeration.png")
