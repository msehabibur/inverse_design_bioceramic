"""The full weight sweep including the published weighting, and the Pareto fronts.

Two gaps the second audit found: Table S15 showed five of the twenty-one swept
weights and omitted the 0.588 the paper actually used, and the NSGA-II front is
described as the primary result of the search without being displayed anywhere.

The sweep needs no model call: the enumeration already carries a predicted
modulus and density for every reachable formula, and the objective is a closed
expression in those two numbers, so the winner at any weight is an argmin over
the table.

Writes 01_data/audit_weight_sweep_full.csv and the Pareto figure.
"""
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "figure_scripts"))
from paths import DATA, ROOT
import paperstyle as ps

TARGET = {"enamel": (85.0, 3.0), "dentin": (20.0, 2.1), "implant": (110.0, 4.4)}
POOL = {"enamel": "enamel", "implant": "enamel", "dentin": "dentin"}
PUBLISHED = 0.588

en = pd.read_csv(os.path.join(DATA, "revision_enumeration_full.csv"))
rows = []
for case, (tE, tR) in TARGET.items():
    s = en[en.pool == POOL[case]]
    dE = (s.pred_modulus_GPa - tE).abs() / tE
    dR = (s.pred_density_gcc - tR).abs() / tR
    for w in list(np.round(np.arange(0.0, 1.0001, 0.05), 3)) + [PUBLISHED]:
        f = w * dE + (1.0 - w) * dR
        i = int(np.argmin(f.values))
        rows.append(dict(case=case, w_modulus=round(float(w), 3),
                         published=bool(abs(w - PUBLISHED) < 1e-9),
                         best_composition=s.composition.iloc[i],
                         best_E_GPa=float(s.pred_modulus_GPa.iloc[i]),
                         best_rho_gcc=float(s.pred_density_gcc.iloc[i]),
                         best_objective=round(float(f.values[i]), 4)))
sw = pd.DataFrame(rows).sort_values(["case", "w_modulus"]).reset_index(drop=True)
sw.to_csv(os.path.join(DATA, "audit_weight_sweep_full.csv"), index=False)
print("wrote 01_data/audit_weight_sweep_full.csv  (%d rows)" % len(sw), flush=True)
for case in TARGET:
    p = sw[(sw.case == case) & sw.published].iloc[0]
    d = sw[(sw.case == case) & ~sw.published]
    print("  %-8s published weighting %.3f -> %s at %.1f GPa; distinct winners over the "
          "sweep: %d" % (case, PUBLISHED, p.best_composition, p.best_E_GPa,
                         d.best_composition.nunique()), flush=True)

# ------------------------------- the exact front over the whole enumerated set
# The NSGA-II run reports the front its population reached. The reachable set is
# finite, so the exact non-dominated set can be computed outright, and for dentin
# it is not the single point the search returned.
ex = []
for case, (tE, tR) in TARGET.items():
    s_ = en[en.pool == POOL[case]].reset_index(drop=True)
    dE = ((s_.pred_modulus_GPa - tE).abs() / tE).values
    dR = ((s_.pred_density_gcc - tR).abs() / tR).values
    best = np.inf
    for i in np.lexsort((dR, dE)):
        if dR[i] < best - 1e-12:
            best = dR[i]
            ex.append(dict(case=case, composition=s_.composition[i],
                           pred_E_GPa=float(s_.pred_modulus_GPa[i]),
                           pred_rho_gcc=float(s_.pred_density_gcc[i]),
                           dist_E=float(dE[i]), dist_rho=float(dR[i])))
exf = pd.DataFrame(ex).sort_values(["case", "dist_E"]).reset_index(drop=True)
exf.to_csv(os.path.join(DATA, "revision_pareto_exact.csv"), index=False)
print("wrote 01_data/revision_pareto_exact.csv  (%s)"
      % ", ".join("%s %d points" % (c, int((exf.case == c).sum())) for c in TARGET),
      flush=True)

# ---------------------------------------------------------------- Pareto front
ps.use()
OUT = os.path.join(ROOT, "04_manuscript", "revision_figures")
fig = plt.figure(figsize=(20.5, 8.2), constrained_layout=True)
gs = fig.add_gridspec(1, 2)

ax = fig.add_subplot(gs[0, 0])
for i, case in enumerate(("enamel", "dentin", "implant")):
    f = os.path.join(DATA, "pareto_%s.csv" % case)
    if not os.path.exists(f):
        continue
    p = pd.read_csv(f).sort_values("dist_E")
    ax.plot(100 * p.dist_E, 100 * p.dist_rho, ps.CASE_MARKER[case],
            color=ps.CASE_COLOR[case], markersize=17, linestyle="-", linewidth=2.4,
            markeredgecolor="white", markeredgewidth=1.2,
            label="%s, search, %d pt" % (ps.tc(ps.CASE_LABEL[case]), len(p)))
    q = exf[exf.case == case].sort_values("dist_E")
    ax.plot(100 * q.dist_E, 100 * q.dist_rho, ps.CASE_MARKER[case],
            color=ps.CASE_COLOR[case], markersize=15, linestyle="--", linewidth=2.0,
            markerfacecolor="none", markeredgewidth=2.4,
            label="%s, exact, %d pt" % (ps.tc(ps.CASE_LABEL[case]), len(q)))
ax.set_xlabel(ps.tc("distance to modulus target (%)"))
ax.set_ylabel(ps.tc("distance to density target (%)"))
ax.margins(0.10)
# six entries over two columns in a band opened above the data, so the legend
# cannot print on top of the fronts it describes
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.005), ncol=2, frameon=False,
          fontsize=ps.BASE - 6, columnspacing=1.4, handletextpad=0.6,
          labelspacing=0.4)
ps.panel_tag(ax, "a", dy=1.30)

ax = fig.add_subplot(gs[0, 1])
for i, case in enumerate(("enamel", "dentin", "implant")):
    s = sw[sw.case == case]
    d = s[~s.published].sort_values("w_modulus")
    ax.plot(d.w_modulus, d.best_E_GPa, ps.CASE_MARKER[case],
            color=ps.CASE_COLOR[case], markersize=14, linestyle="-", linewidth=2.6,
            label=ps.tc(ps.CASE_LABEL[case]))
    p = s[s.published].iloc[0]
    # the published weighting as an open ring around its point: circles only
    ax.plot([p.w_modulus], [p.best_E_GPa], "o", markersize=30, markerfacecolor="none",
            markeredgecolor=ps.COLORS[3], markeredgewidth=3.4, linestyle="none",
            label="Published Weighting" if i == 0 else None)
    ax.axhline(TARGET[case][0], color=ps.CASE_COLOR[case], linewidth=1.8,
               linestyle=":")
ax.set_xlabel(ps.tc("modulus weight"))
ax.set_ylabel(ps.tc("best enumerated modulus (GPa)"))
ps.legend_above(ax, ncol=2)
ps.panel_tag(ax, "b", dy=1.30)

print("wrote", ps.save(fig, "FigureR6_front_and_weights", OUT), flush=True)
print("SWEEP_PARETO_DONE", flush=True)
