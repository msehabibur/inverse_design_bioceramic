"""Revision figures, in the manuscript figure style (Arial Narrow, large type,
compact symmetric panels, distinguishable line styles, error bars).

  FigureR1  split-protocol validation of the chained surrogate
  FigureR2  modulus and aqueous chemistry of the decomposition mixtures
  FigureR3  random-search control and enumeration at the decoder bounds

Run after revision_validation.py, revision_mixture_and_chemistry.py and
revision_search_baseline.py. FigureR3 is skipped if its input is absent.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import paperstyle as ps
from paths import DATA, MANUSCRIPT

ps.use()
OUT = os.path.join(MANUSCRIPT, "revision_figures")
TARGET = {"enamel": 85.0, "dentin": 20.0, "implant": 110.0}
SPLIT_LABEL = {"random": "random folds",
               "formula": "folds grouped by reduced formula",
               "chemsys": "folds grouped by chemical system"}


def sub(formula):
    """Chemical formula with mathtext subscripts, e.g. Ca3(PO4)2."""
    out, i = "", 0
    while i < len(formula):
        ch = formula[i]
        if ch.isdigit():
            j = i
            while j < len(formula) and formula[j].isdigit():
                j += 1
            out += "$_{%s}$" % formula[i:j]
            i = j
        else:
            out += ch
            i += 1
    return out


def group_names(fig, ax, xs, pools, pool_name, per):
    """Pool names under the upright tick labels, a fixed number of points below the
    tallest of them, so they read as the group label of their own panel and not
    as a caption of the panel underneath."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    h = max(t.get_window_extent(r).height for t in ax.get_xticklabels() if t.get_text())
    drop = h * 72.0 / fig.dpi + plt.rcParams["xtick.major.pad"] + 10.0
    for k, pool in enumerate(pools):
        ax.annotate(pool_name[pool], xy=(xs[k * per + 1], 0.0),
                    xycoords=("data", "axes fraction"), xytext=(0, -drop),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=ps.BASE - 2)


def figure_r1():
    v = pd.read_csv(os.path.join(DATA, "revision_validation.csv"))
    mod = v[v.target == "youngs_modulus_GPa"]
    old = pd.read_csv(os.path.join(DATA, "uncertainty_intervals.csv"))
    new = pd.read_csv(os.path.join(DATA, "revision_uncertainty.csv"))

    fig = plt.figure(figsize=(20.5, 15.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2)

    # (a) coefficient of determination, both pools, three splits, two modes
    ax = fig.add_subplot(gs[0, 0])
    pools = ["enamel_implant", "dentin"]
    pool_name = {"enamel_implant": "Enamel and Implant Pool",
                 "dentin": "Dentin Pool"}
    splits = ["random", "formula", "chemsys"]
    w = 0.35
    xs = np.arange(len(splits) * len(pools), dtype=float)
    xs[len(splits):] += 0.9
    for j, mode in enumerate(("true", "design")):
        vals, pos = [], []
        for i, (pool, sp) in enumerate([(p, s) for p in pools for s in splits]):
            row = mod[(mod.pool == pool) & (mod.split == sp) & (mod.features == mode)]
            vals.append(float(row.r2.iloc[0]))
            pos.append(xs[i] + (j - 0.5) * w)
        ax.bar(pos, vals, width=w,
               color=ps.COLORS[0] if mode == "true" else ps.COLORS[1],
               edgecolor="white", linewidth=1.2,
               hatch=None if mode == "true" else "//",
               label=ps.tc("database density and volume") if mode == "true"
               else ps.tc("predicted density and volume"))
        for p, val in zip(pos, vals):
            ax.text(p, val + 0.015, f"{val:.2f}", ha="center", va="bottom",
                    rotation=90, fontsize=ps.BASE - 6)
    ax.set_xticks(xs)
    ax.set_xticklabels(["Random", "Formula", "Chemistry"] * 2, rotation=90,
                       fontsize=ps.BASE - 6)
    ax.set_ylabel(ps.tc("coefficient of determination"))
    ax.set_ylim(0, 1.12)
    ax.set_xlim(xs[0] - 0.75, xs[-1] + 0.75)
    ps.legend_above(ax)
    group_names(fig, ax, xs, pools, pool_name, len(splits))
    ps.panel_tag(ax, "a")

    # (b) mean absolute error, same layout
    ax = fig.add_subplot(gs[0, 1])
    for j, mode in enumerate(("true", "design")):
        vals, pos = [], []
        for i, (pool, sp) in enumerate([(p, s) for p in pools for s in splits]):
            row = mod[(mod.pool == pool) & (mod.split == sp) & (mod.features == mode)]
            vals.append(float(row.mae.iloc[0]))
            pos.append(xs[i] + (j - 0.5) * w)
        ax.bar(pos, vals, width=w,
               color=ps.COLORS[0] if mode == "true" else ps.COLORS[1],
               edgecolor="white", linewidth=1.2,
               hatch=None if mode == "true" else "//")
        for p, val in zip(pos, vals):
            ax.text(p, val + 0.7, f"{val:.0f}", ha="center", va="bottom",
                    rotation=90, fontsize=ps.BASE - 6)
    ax.set_xticks(xs)
    ax.set_xticklabels(["Random", "Formula", "Chemistry"] * 2, rotation=90,
                       fontsize=ps.BASE - 6)
    ax.set_ylabel(ps.tc("mean absolute error (GPa)"))
    ax.set_ylim(0, 62)
    ax.set_xlim(xs[0] - 0.75, xs[-1] + 0.75)
    group_names(fig, ax, xs, pools, pool_name, len(splits))
    ps.panel_tag(ax, "b")

    # (c) interval width by protocol
    ax = fig.add_subplot(gs[1, 0])
    for i, pool in enumerate(pools):
        lo = [float(mod[(mod.pool == pool) & (mod.split == s)
                        & (mod.features == "true")].pi_lo_mult.iloc[0]) for s in splits] + \
             [float(mod[(mod.pool == pool) & (mod.split == s)
                        & (mod.features == "design")].pi_lo_mult.iloc[0]) for s in splits]
        hi = [float(mod[(mod.pool == pool) & (mod.split == s)
                        & (mod.features == "true")].pi_hi_mult.iloc[0]) for s in splits] + \
             [float(mod[(mod.pool == pool) & (mod.split == s)
                        & (mod.features == "design")].pi_hi_mult.iloc[0]) for s in splits]
        x = np.arange(6) + (i - 0.5) * 0.3
        ax.errorbar(x, np.ones(6), yerr=[np.ones(6) - lo, np.array(hi) - 1],
                    fmt=ps.MARKERS[i], color=ps.COLORS[i], markersize=13,
                    linewidth=3.0, capsize=8, capthick=3.0,
                    label=ps.tc(pool_name[pool]))
    ax.axhline(1.0, color="0.35", linewidth=1.6, linestyle=":")
    ax.set_xticks(np.arange(6))
    ax.set_xticklabels(["Random\nDatabase", "Formula\nDatabase", "Chemistry\nDatabase",
                        "Random\nPredicted", "Formula\nPredicted", "Chemistry\nPredicted"],
                       fontsize=ps.BASE - 6)
    ax.set_ylabel(ps.tc("multiplicative interval"))
    ax.set_ylim(0.34, 2.62)
    ps.legend_above(ax, ncol=2)
    ps.panel_tag(ax, "c")

    # (d) dentin intervals, published protocol against the design-time protocol
    ax = fig.add_subplot(gs[1, 1])
    o = old[old.showcase == "dentin"].sort_values("rank")
    n = new[new.showcase == "dentin"].sort_values("rank")
    x = np.arange(len(o)) + 1
    ax.errorbar(x - 0.16, o.pred_E_GPa, fmt=ps.MARKERS[0], color=ps.COLORS[0],
                yerr=[o.pred_E_GPa - o.E_lo_GPa, o.E_hi_GPa - o.pred_E_GPa],
                linestyle="none", markersize=15, linewidth=3.0, capsize=7,
                capthick=3.0, label=ps.tc("database-input interval"))
    ax.errorbar(x + 0.16, n.pred_E_GPa, fmt=ps.MARKERS[1], color=ps.COLORS[1],
                yerr=[n.pred_E_GPa - n.E_lo_GPa, n.E_hi_GPa - n.pred_E_GPa],
                linestyle="none", markersize=15, linewidth=3.0, capsize=7,
                capthick=3.0, label=ps.tc("design-time interval"))
    ax.axhline(20.0, color=ps.COLORS[2], linewidth=3.0, linestyle="--")
    ax.text(0.7, 11.0, "Dentin Target, 20 GPa", color=ps.COLORS[2],
            fontsize=ps.BASE - 4, va="center", ha="left")
    ax.set_xlabel(ps.tc("dentin candidate"))
    ax.set_ylabel(ps.tc("Young's modulus (GPa)"))
    ax.set_xticks(x)
    ax.set_ylim(0, 138)
    ax.legend(loc="upper left", fontsize=ps.BASE - 6)
    ps.panel_tag(ax, "d")

    p = ps.save(fig, "FigureR1_validation_protocols", OUT)
    print("wrote", p, flush=True)


def figure_r2():
    m = pd.read_csv(os.path.join(DATA, "revision_mixture_modulus.csv"))
    ph = pd.read_csv(os.path.join(DATA, "revision_phase_moduli.csv"))

    fig = plt.figure(figsize=(28.0, 8.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 3)

    # (a) single phase against mixture, per candidate
    ax = fig.add_subplot(gs[0, 0])
    for i, sc in enumerate(("enamel", "dentin", "implant")):
        s = m[m.showcase == sc]
        ax.errorbar(s.pred_E_single_phase_GPa, s.E_hill_GPa,
                    yerr=[s.E_hill_GPa - s.E_reuss_GPa,
                          s.E_voigt_GPa - s.E_hill_GPa],
                    fmt=ps.CASE_MARKER[sc], color=ps.CASE_COLOR[sc],
                    linestyle="none", markersize=16, linewidth=2.6, capsize=6,
                    capthick=2.6, label=ps.tc(ps.CASE_LABEL[sc]))
    ax.plot([0, 175], [0, 175], color="0.35", linewidth=1.8, linestyle=":")
    # pad the low side so markers near zero are not cut by the spines
    lim = [-8, 175]
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel(ps.tc("composition-model prediction (GPa)"))
    ax.set_ylabel(ps.tc("decomposition mixture (GPa)"))
    ax.legend(loc="lower right")
    ps.panel_tag(ax, "a")

    # (b) mixture modulus against each target
    ax = fig.add_subplot(gs[0, 1])
    ax.set_ylim(0, 205)
    for i, sc in enumerate(("enamel", "dentin", "implant")):
        s = m[m.showcase == sc].sort_values("rank")
        x = np.arange(len(s)) + 1
        ax.errorbar(x + (i - 1) * 0.22, s.E_hill_GPa,
                    yerr=[s.E_hill_GPa - s.E_reuss_GPa,
                          s.E_voigt_GPa - s.E_hill_GPa],
                    fmt=ps.CASE_MARKER[sc], color=ps.CASE_COLOR[sc],
                    linestyle="none", markersize=14, linewidth=2.4, capsize=5,
                    capthick=2.4, label=ps.tc(ps.CASE_LABEL[sc]))
        ln = ax.axhline(TARGET[sc], color=ps.CASE_COLOR[sc], linewidth=2.6)
        ln.set_dashes(list(ps.CASE_DASH[sc]) if ps.CASE_DASH[sc][0] else [8, 0])
    ax.set_xlabel(ps.tc("candidate, ranked by objective value"))
    ax.set_ylabel(ps.tc("mixture Young's modulus (GPa)"))
    ax.set_xticks(np.arange(1, 16, 2))
    ps.legend_above(ax, ncol=3, handlelength=1.4)
    ps.panel_tag(ax, "b")

    # (c) soluble alkali fraction of the dentin decompositions
    ax = fig.add_subplot(gs[0, 2])
    s = m[m.showcase == "dentin"].sort_values("rank")
    ax.bar(np.arange(len(s)) + 1, s.soluble_vol_frac, width=0.68,
           color=ps.CASE_COLOR["dentin"], edgecolor="white", linewidth=1.2)
    ax.axhline(float(s.soluble_vol_frac.median()), color="0.30",
               linewidth=2.4, linestyle=":",
               label=f"Median, {s.soluble_vol_frac.median():.2f}")
    ps.legend_above(ax)
    ax.set_xlabel(ps.tc("dentin candidate"))
    ax.set_ylabel(ps.tc("potentially soluble phase volume fraction"))
    ax.set_xticks(np.arange(1, 16, 2))
    ax.set_ylim(0, 1.06)
    ps.panel_tag(ax, "c")

    p = ps.save(fig, "FigureR2_mixture_and_solubility", OUT)
    print("wrote", p, flush=True)

    # separate, taller panel for the 42 product phases
    fig = plt.figure(figsize=(13.0, 20.0), constrained_layout=True)
    ax = fig.add_subplot(111)
    ph = ph.sort_values("E")
    isdb = (ph.E_source != "surrogate").values
    y = np.arange(len(ph))
    ax.barh(y[isdb], ph.E.values[isdb], color=ps.COLORS[0], height=0.74,
            label=ps.tc("database elastic tensor"))
    ax.barh(y[~isdb], ph.E.values[~isdb], color=ps.COLORS[1], height=0.74,
            hatch="//", edgecolor="white", linewidth=0.8,
            label=ps.tc("surrogate, database structure"))
    ax.axvline(20.0, color=ps.COLORS[2], linewidth=3.0, linestyle="--",
               label="Dentin Target, 20 GPa")
    ax.set_yticks(y)
    ax.set_yticklabels([sub(f) for f in ph.formula], fontsize=ps.BASE - 6)
    ax.set_ylim(-0.8, len(ph) - 0.2)
    ax.set_xlabel(ps.tc("Young's modulus of the product phase (GPa)"))
    ax.legend(loc="lower right", fontsize=ps.BASE - 6)
    p = ps.save(fig, "FigureR2b_product_phase_moduli", OUT)
    print("wrote", p, flush=True)


def figure_r3():
    f = os.path.join(DATA, "revision_search_baseline.csv")
    g = os.path.join(DATA, "revision_enumeration_full.csv")
    if not (os.path.exists(f) and os.path.exists(g)):
        print("FigureR3 skipped: search-baseline output not present yet", flush=True)
        return
    bl = pd.read_csv(f)
    en = pd.read_csv(g)

    fig = plt.figure(figsize=(20.5, 8.2), constrained_layout=True)
    gs = fig.add_gridspec(1, 2)

    # (a) genetic algorithm against random search at equal budget
    ax = fig.add_subplot(gs[0, 0])
    cases = ["enamel", "dentin", "implant"]
    x = np.arange(len(cases), dtype=float)
    rand_mean = [bl[bl.case == c].best_fitness.mean() for c in cases]
    rand_sd = [bl[bl.case == c].best_fitness.std() for c in cases]
    ga = [bl[bl.case == c].ga_best_fitness.iloc[0] for c in cases]
    ax.bar(x - 0.2, ga, width=0.4, color=ps.COLORS[0], edgecolor="white",
           linewidth=1.2, label=ps.tc("genetic algorithm"))
    ax.bar(x + 0.2, rand_mean, width=0.4, yerr=rand_sd, color=ps.COLORS[1],
           edgecolor="white", linewidth=1.2, hatch="//", capsize=8,
           error_kw=dict(elinewidth=3.0, capthick=3.0),
           label=ps.tc("random search, same budget"))
    # the random bar carries an error bar, so its label clears the whisker cap
    for xi, g, rm, sd in zip(x, ga, rand_mean, rand_sd):
        ax.text(xi - 0.2, g + 0.03, f"{g:.3f}", ha="center", va="bottom",
                fontsize=ps.BASE - 6)
        ax.text(xi + 0.2, rm + (0.0 if np.isnan(sd) else sd) + 0.05,
                f"{rm:.3f}", ha="center", va="bottom", fontsize=ps.BASE - 6)
    ax.set_xticks(x)
    ax.set_xticklabels([ps.tc(ps.CASE_LABEL[c]) for c in cases])
    ax.set_ylabel(ps.tc("best objective value reached"))
    ax.set_ylim(0, 1.18)
    ps.legend_above(ax)
    ps.panel_tag(ax, "a")

    # (b) enumerated modulus distribution at the decoder bounds
    ax = fig.add_subplot(gs[0, 1])
    for pool in ("enamel", "dentin"):
        if pool not in set(en.pool.unique()):
            continue
        s = en[en.pool == pool]
        name = "enamel and implant" if pool == "enamel" else "dentin"
        ax.hist(s.pred_modulus_GPa, bins=60, histtype="step", linewidth=3.0,
                color=ps.CASE_COLOR[pool],
                label=f"{ps.tc(name)} Pool, {len(s):,} Formulas, "
                      f"Minimum Prediction {s.pred_modulus_GPa.min():.1f} GPa")
    ax.axvline(20.0, color=ps.COLORS[2], linewidth=3.0, linestyle="--")
    ax.text(22, ax.get_ylim()[1] * 0.72, "Dentin Target", color=ps.COLORS[2],
            fontsize=ps.BASE - 4, rotation=90, va="center")
    ax.set_xlabel(ps.tc("predicted Young's modulus (GPa)"))
    ax.set_ylabel(ps.tc("number of formulas"))
    ax.set_xlim(8, float(en.pred_modulus_GPa.max()) * 1.04)
    ax.set_ylim(0, ax.get_ylim()[1] * 1.12)
    ps.legend_above(ax, handlelength=2.0)
    ps.panel_tag(ax, "b")

    p = ps.save(fig, "FigureR3_search_and_enumeration", OUT)
    print("wrote", p, flush=True)


def figure_r4():
    f = os.path.join(DATA, "revision_sampling.csv")
    g = os.path.join(DATA, "revision_sampling_summary.csv")
    if not (os.path.exists(f) and os.path.exists(g)):
        print("FigureR4 skipped: sampling output not present yet", flush=True)
        return
    samp = pd.read_csv(f)
    sm = pd.read_csv(g)

    fig = plt.figure(figsize=(28.0, 8.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 3)

    # (a) how many relaxations reached the force criterion
    ax = fig.add_subplot(gs[0, 0])
    # the fraction is over the trials that returned an energy, the denominator
    # Table S14 states, so the bar and the table print the same percentage
    by = samp.groupby("model").agg(tr=("n_energies", "sum"),
                                   cv=("n_converged", "sum"),
                                   sp=("spread_eV_atom", "median")).reset_index()
    by = by.sort_values("cv", ascending=False)
    frac = 100.0 * by.cv / by.tr
    x = np.arange(len(by))
    ax.bar(x, frac, width=0.62, color=ps.COLORS[0], edgecolor="white", linewidth=1.2)
    for xi, v, c, t in zip(x, frac, by.cv, by.tr):
        ax.text(xi, v + 1.2, f"{int(c)} of {int(t)}", ha="center", va="bottom",
                fontsize=ps.BASE - 6)
    proper = {"mace": "MACE", "mattersim": "MatterSim", "m3gnet": "M3GNet",
              "sevennet": "SevenNet", "chgnet": "CHGNet"}
    ax.set_xticks(x)
    ax.set_xticklabels([proper.get(m, m) for m in by.model], fontsize=ps.BASE - 6)
    ax.set_ylabel(ps.tc("converged trials (percent)"))
    ax.set_ylim(0, 45)
    ps.panel_tag(ax, "a")

    # (b) energy above hull against the spread the sampling shows
    ax = fig.add_subplot(gs[0, 1])
    for sc in ("enamel", "dentin", "implant"):
        t = sm[sm.showcase == sc]
        ax.plot(t.median_spread, t.ehull_median, ps.CASE_MARKER[sc],
                color=ps.CASE_COLOR[sc], markersize=16, linestyle="none",
                label=ps.tc(ps.CASE_LABEL[sc]))
    lim = [0.15, 6.0]
    # the dashed line is the equality spread = energy above hull, and nothing more:
    # the spread across eight trials is not a bound on how much lower an unsampled
    # structure could lie
    ax.plot(lim, lim, color="0.30", linewidth=2.4, linestyle="--",
            label="Spread Equals Energy Above Hull")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*lim); ax.set_ylim(0.1, 6.0)
    ax.set_xlabel(ps.tc("median trial spread (eV/atom)"))
    ax.set_ylabel(ps.tc("energy above hull (eV/atom)"))
    ps.legend_above(ax, ncol=2)
    ps.panel_tag(ax, "b")

    # (c) how much of the required improvement the sampling noise already covers
    ax = fig.add_subplot(gs[0, 2])
    ax.hist(sm.margin_over_spread, bins=22, color=ps.COLORS[1],
            edgecolor="white", linewidth=1.2)
    ax.axvline(1.0, color="0.20", linewidth=3.0, linestyle="--")
    ax.text(1.08, ax.get_ylim()[1] * 0.82,
            f"{int((sm.margin_over_spread < 1).sum())} of {len(sm)} Candidates\n"
            f"Fall Below One", fontsize=ps.BASE - 6, va="top")
    ax.set_xlabel(ps.tc("required margin, in units of the spread"))
    ax.set_ylabel(ps.tc("number of candidates"))
    ax.set_xlim(0, 1.65)
    ax.set_xticks([0.5, 1.0, 1.5])
    ps.panel_tag(ax, "c")

    p = ps.save(fig, "FigureR4_sampling_convergence", OUT)
    print("wrote", p, flush=True)


def figure_r5():
    """Where the interval is reliable and where it is not.

    The modulus model carries a marginal coverage close to nominal, but the
    dentin conclusion rests on the soft end of the range, so the figure shows
    the error and the coverage conditioned on the measured modulus, and then the
    interval applied to the enumerated dentin space.
    """
    f1 = os.path.join(DATA, "audit_oof_enamel_implant.csv")
    f2 = os.path.join(DATA, "audit_oof_dentin.csv")
    f3 = os.path.join(DATA, "revision_enumeration_full.csv")
    if not all(os.path.exists(x) for x in (f1, f2, f3)):
        print("FigureR5 skipped: audit output not present yet", flush=True)
        return
    a = pd.read_csv(f1)
    b = pd.read_csv(f2)
    en = pd.read_csv(f3)
    en = en[en.pool == "dentin"]

    fig = plt.figure(figsize=(28.0, 8.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 3)

    # (a) measured against out-of-fold prediction, both pools
    ax = fig.add_subplot(gs[0, 0])
    keys = []
    for i, (d, nm) in enumerate(((a, "enamel and implant pool"),
                                 (b, "dentin pool"))):
        ax.plot(d.youngs_modulus_GPa, d.pred_E_GPa, ps.MARKERS[i],
                color=ps.COLORS[i], markersize=11, markeredgecolor='white', markeredgewidth=0.8, linestyle="none")
        keys.append(Line2D([], [], marker=ps.MARKERS[i], linestyle="none",
                           markersize=13, markerfacecolor=ps.COLORS[i],
                           markeredgecolor="white", markeredgewidth=1.0,
                           label=ps.tc(nm)))
    lim = [4.0, 600.0]
    ax.plot(lim, lim, color="0.30", linewidth=2.2, linestyle=":")
    ax.axvline(40.0, color=ps.COLORS[2], linewidth=2.6, linestyle="--")
    ax.text(43, 8.0, "40 GPa", color=ps.COLORS[2], fontsize=ps.BASE - 6,
            rotation=90, va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(*lim); ax.set_ylim(*lim)
    ax.set_xlabel(ps.tc("database modulus (GPa)"))
    ax.set_ylabel(ps.tc("out-of-fold prediction (GPa)"))
    ps.legend_above(ax, handles=keys, ncol=2)
    ps.panel_tag(ax, "a")

    # (b) coverage of the 90 percent interval by measured-modulus quintile
    ax = fig.add_subplot(gs[0, 1])
    w = 0.38
    for i, (d, nm) in enumerate(((a, "enamel and implant pool"),
                                 (b, "dentin pool"))):
        q = pd.qcut(d.youngs_modulus_GPa, 5, labels=False, duplicates="drop")
        cov, cen = [], []
        for k in sorted(pd.unique(q.dropna())):
            g = d[q == k]
            cov.append(float(((g.E_lo_GPa <= g.youngs_modulus_GPa)
                              & (g.youngs_modulus_GPa <= g.E_hi_GPa)).mean()))
            cen.append(float(g.youngs_modulus_GPa.median()))
        x = np.arange(len(cov)) + (i - 0.5) * w
        ax.bar(x, cov, width=w, color=ps.COLORS[i], edgecolor="white",
               linewidth=1.2, hatch=None if i == 0 else "//",
               label=ps.tc(nm))
        # the value goes inside its own bar, on its side: two labels above a pair
        # of bars touch at this type size however they are nudged
        # at the foot of its own bar, well below the 0.90 line it would otherwise cross
        for xi, c in zip(x, cov):
            ax.text(xi, 0.03, f"{c:.2f}", ha="center", va="bottom", gid="inside",
                    rotation=90, color="white", fontsize=ps.BASE - 6)
        if i == 0:
            ticks = ["%.0f" % c for c in cen]
    ax.axhline(0.90, color="0.20", linewidth=2.6, linestyle="--", label="Nominal, 0.90")
    ax.set_xticks(np.arange(len(ticks)))
    ax.set_xticklabels(ticks)
    ax.set_xlabel(ps.tc("database modulus quintile (GPa)"))
    ax.set_ylabel(ps.tc("interval coverage"))
    ax.set_ylim(0, 1.46)
    ax.legend(loc="upper left", fontsize=ps.BASE - 6)
    ps.panel_tag(ax, "b")

    # (c) the interval carried across the enumerated dentin space
    ax = fig.add_subplot(gs[0, 2])
    v = pd.read_csv(os.path.join(DATA, "revision_validation.csv"))
    qq = float(v[(v.pool == "dentin") & (v.target == "youngs_modulus_GPa")
                 & (v.split == "chemsys") & (v.features == "design")].conformal_q.iloc[0])
    e = en.sort_values("pred_modulus_GPa").reset_index(drop=True)
    x = np.arange(1, 51)
    p = e.pred_modulus_GPa.values[:50]
    ax.errorbar(x, p, yerr=[p - p * 10 ** -qq, p * 10 ** qq - p], fmt="o",
                color=ps.CASE_COLOR["dentin"], markersize=16, linestyle="none",
                elinewidth=2.4, capsize=5, capthick=2.4,
                label=ps.tc("90 percent interval"))
    ax.axhline(20.0, color=ps.COLORS[2], linewidth=3.0, linestyle="--",
               label="Dentin Target, 20 GPa")
    # the count is read from the table that supplied the points, never typed
    ax.set_xlabel(ps.tc("fifty lowest of %s reachable formulas"
                        % format(len(en), ",")))
    ax.set_ylabel(ps.tc("predicted modulus (GPa)"))
    ax.set_ylim(0, float((p * 10 ** qq).max()) * 1.34)
    ax.legend(loc="upper left", fontsize=ps.BASE - 6)
    ps.panel_tag(ax, "c")

    p_ = ps.save(fig, "FigureR5_interval_reliability", OUT)
    print("wrote", p_, flush=True)


figure_r1()
figure_r2()
figure_r3()
figure_r4()
figure_r5()
print("REVISION_FIGURES_DONE", flush=True)
