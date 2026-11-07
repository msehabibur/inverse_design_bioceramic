"""Revision audit v2 - the calculation-level checks asked for before resubmission.

Six checks, each printing what it measured and writing a CSV:

  A  what the consensus decomposition coefficients actually are. The shipped
     analysis called them mole fractions. pymatgen returns barycentric
     coordinates in atom-fraction space, so the test reconstructs each
     candidate under both readings and reports which one closes.
  B  the mixture quantities recomputed with the volume conversion that matches
     the coefficient definition, against the shipped values.
  C  the five-model consensus recomputed with the quality exclusion the
     supplementary table declares, that is, without the failed negative
     energies.
  D  the design-time interval applied to the lowest-predicted formulas of the
     enumerated dentin space, not only to the fifteen the search returned.
  E  coverage of that interval measured on data that did not set the quantile,
     through a nested grouped protocol.
  F  a low-modulus holdout: every pool entry below 40 GPa removed from
     training, then predicted, which is the regime the dentin conclusion rests on.

Writes 01_data/audit_*.csv.
"""
import os, re, sys
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import DATA, ROOT
from chain import POOLS, Chain, group_folds, load_pool, released

AMU = 1.66054
COVERAGE = 0.90
NFOLD = 5
SEED = 42
TARGET = {"enamel": 85.0, "dentin": 20.0, "implant": 110.0}
MODELS = ["chgnet", "m3gnet", "mace", "mattersim", "omni"]


def gbr():
    return HistGradientBoostingRegressor(random_state=SEED)


def parse_decomp(s):
    out = []
    for tok in str(s).split(" + "):
        m = re.match(r"^(.*?)\s*\(([0-9]*\.?[0-9]+)\)$", tok.strip())
        if m:
            out.append((m.group(1), float(m.group(2))))
    return out


def atom_frac(comp):
    d = comp.get_el_amt_dict()
    n = sum(d.values())
    return {k: v / n for k, v in d.items()}


def dev(a, b):
    keys = set(a) | set(b)
    return max(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


st = pd.read_csv(os.path.join(ROOT, "06_mlff_stability", "stability_mlff.csv"))
ph = pd.read_csv(os.path.join(DATA, "revision_phase_moduli.csv"))
info = {r["formula"]: r for _, r in ph.iterrows()}
shipped = pd.read_csv(os.path.join(DATA, "revision_mixture_modulus.csv"))
val = pd.read_csv(os.path.join(DATA, "revision_validation.csv"))

print("=" * 78)
print("A  what the decomposition coefficients are")
print("=" * 78)
arows = []
for _, r in st.iterrows():
    dec = parse_decomp(r["decomposition_consensus"])
    if not dec:
        continue
    c = np.array([v for _, v in dec], float)
    c = c / c.sum()
    tgt = atom_frac(Composition(r["composition"]))
    # reading 1: the coefficients are fractions of atoms
    mix_atom = {}
    for (f, _), ci in zip(dec, c):
        for k, v in atom_frac(Composition(f)).items():
            mix_atom[k] = mix_atom.get(k, 0.0) + ci * v
    # reading 2: the coefficients are formula-unit mole fractions
    tot = {}
    for (f, _), ci in zip(dec, c):
        for k, v in Composition(f).get_el_amt_dict().items():
            tot[k] = tot.get(k, 0.0) + ci * v
    n = sum(tot.values())
    mix_fu = {k: v / n for k, v in tot.items()}
    arows.append(dict(showcase=r["showcase"], rank=r["rank"],
                      composition=r["composition"],
                      dev_if_atom_fraction=round(dev(tgt, mix_atom), 5),
                      dev_if_formula_unit_fraction=round(dev(tgt, mix_fu), 5)))
A = pd.DataFrame(arows)
A.to_csv(os.path.join(DATA, "audit_coefficient_meaning.csv"), index=False)
print(f"  {len(A)} candidates tested")
print(f"  read as atom fractions        : max deviation over all candidates "
      f"{A.dev_if_atom_fraction.max():.2e}, median {A.dev_if_atom_fraction.median():.2e}")
print(f"  read as formula-unit fractions: max deviation over all candidates "
      f"{A.dev_if_formula_unit_fraction.max():.2e}, median "
      f"{A.dev_if_formula_unit_fraction.median():.2e}")
print(f"  candidates closing to 1e-6 as atom fractions: "
      f"{int((A.dev_if_atom_fraction < 1e-6).sum())}/{len(A)}")
print(f"  candidates closing to 1e-6 as formula-unit fractions: "
      f"{int((A.dev_if_formula_unit_fraction < 1e-6).sum())}/{len(A)}")

print()
print("=" * 78)
print("B  the mixture quantities with the matching volume conversion")
print("=" * 78)
SOLUBLE = {"K3PO4", "Na3PO4", "K2O", "Na2O", "K2Si2O5", "K2Si4O9",
           "Na2Si2O5", "K4P2O7", "Na4P2O7", "KPO3", "NaPO3", "K2CaP2O7"}
brows = []
for _, r in st.iterrows():
    dec = parse_decomp(r["decomposition_consensus"])
    if not dec or any(f not in info for f, _ in dec):
        continue
    a = np.array([v for _, v in dec], float)
    a = a / a.sum()                                    # fractions of atoms
    nat = np.array([Composition(f).num_atoms for f, _ in dec], float)
    vpa = np.array([float(info[f]["vol_per_atom"]) for f, _ in dec])
    Vfu = np.array([Composition(f).weight / float(info[f]["density"]) for f, _ in dec])
    E = np.array([float(info[f]["E"]) for f, _ in dec])
    sol = np.array([f in SOLUBLE for f, _ in dec], bool)

    nfu = a / nat
    nfu = nfu / nfu.sum()                              # formula-unit mole fractions
    vf = a * vpa
    vf = vf / vf.sum()                                 # correct volume fractions
    vf_old = a * Vfu
    vf_old = vf_old / vf_old.sum()                     # the shipped conversion

    def vrh(w):
        v = float((w * E).sum())
        rr = float(1.0 / (w / E).sum())
        return v, rr, 0.5 * (v + rr)

    v_n, r_n, h_n = vrh(vf)
    v_o, r_o, h_o = vrh(vf_old)
    t = TARGET[r["showcase"]]
    brows.append(dict(
        showcase=r["showcase"], rank=int(r["rank"]), composition=r["composition"],
        n_phases=len(dec),
        n_phases_mp_elastic=int(sum(1 for f, _ in dec
                                    if info[f]["E_source"] != "surrogate")),
        atom_frac_soluble=round(float(a[sol].sum()), 3),
        mole_frac_soluble=round(float(nfu[sol].sum()), 3),
        vol_frac_soluble=round(float(vf[sol].sum()), 3),
        vol_frac_soluble_shipped=round(float(vf_old[sol].sum()), 3),
        E_voigt_GPa=round(v_n, 1), E_reuss_GPa=round(r_n, 1), E_hill_GPa=round(h_n, 1),
        E_voigt_shipped=round(v_o, 1), E_reuss_shipped=round(r_o, 1),
        E_hill_shipped=round(h_o, 1),
        target_E=t,
        target_within_bounds=bool(r_n <= t <= v_n),
        target_within_bounds_shipped=bool(r_o <= t <= v_o),
        soluble_phases="+".join(f for (f, _), s in zip(dec, sol) if s),
        decomposition=r["decomposition_consensus"]))
B = pd.DataFrame(brows)
B.to_csv(os.path.join(DATA, "audit_mixture_v2.csv"), index=False)
print(f"  {len(B)}/45 candidates have complete phase data")
for sc in ("enamel", "dentin", "implant"):
    s = B[B.showcase == sc]
    if not len(s):
        continue
    print(f"  {sc:8s} target {TARGET[sc]:5.0f} GPa")
    print(f"    Hill  corrected {s.E_hill_GPa.min():6.1f}-{s.E_hill_GPa.max():6.1f} "
          f"(median {s.E_hill_GPa.median():6.1f})   shipped "
          f"{s.E_hill_shipped.min():6.1f}-{s.E_hill_shipped.max():6.1f} "
          f"(median {s.E_hill_shipped.median():6.1f})")
    print(f"    Reuss floor corrected {s.E_reuss_GPa.min():6.1f}   shipped "
          f"{s.E_reuss_shipped.min():6.1f}")
    print(f"    target inside [Reuss, Voigt]: corrected "
          f"{int(s.target_within_bounds.sum())}/{len(s)}   shipped "
          f"{int(s.target_within_bounds_shipped.sum())}/{len(s)}")
    print(f"    soluble fraction of the assembly: atoms "
          f"{s.atom_frac_soluble.median():.2f}, formula units "
          f"{s.mole_frac_soluble.median():.2f}, volume {s.vol_frac_soluble.median():.2f} "
          f"(shipped volume {s.vol_frac_soluble_shipped.median():.2f})")
    print(f"    phases with a database elastic tensor: "
          f"{int(s.n_phases_mp_elastic.sum())} of {int(s.n_phases.sum())} phase slots")

print()
print("=" * 78)
print("C  the consensus with the declared quality exclusion")
print("=" * 78)
crows = []
for _, r in st.iterrows():
    vals = {m: r[m] for m in MODELS if m in st.columns and pd.notna(r[m])}
    good = {m: v for m, v in vals.items() if v > 0}
    crows.append(dict(showcase=r["showcase"], rank=int(r["rank"]),
                      composition=r["composition"],
                      n_models=len(vals), n_valid=len(good),
                      n_negative=len(vals) - len(good),
                      median_shipped=round(float(np.median(list(vals.values()))), 4),
                      median_valid_only=round(float(np.median(list(good.values()))), 4)
                      if good else np.nan))
C = pd.DataFrame(crows)
C["delta"] = (C.median_valid_only - C.median_shipped).round(4)
C.to_csv(os.path.join(DATA, "audit_stability_v2.csv"), index=False)
ch = C[C.delta.abs() > 1e-6]
print(f"  candidates with at least one negative model energy: "
      f"{int((C.n_negative > 0).sum())}/{len(C)}")
print(f"  candidates whose consensus median moves once those are excluded: {len(ch)}")
print(f"  largest move: {C.delta.abs().max():.3f} eV/atom")
print(f"  median above 0.05 eV/atom, shipped    : "
      f"{int((C.median_shipped > 0.05).sum())}/{len(C)}")
print(f"  median above 0.05 eV/atom, valid only : "
      f"{int((C.median_valid_only > 0.05).sum())}/{len(C)}")
if len(ch):
    print("  the moved candidates:")
    for _, r in ch.iterrows():
        print(f"    {r['composition']:26s} {r['median_shipped']:6.3f} -> "
              f"{r['median_valid_only']:6.3f}  ({int(r['n_valid'])} valid of "
              f"{int(r['n_models'])})")

print()
print("=" * 78)
print("D  the design-time interval over the enumerated dentin space")
print("=" * 78)
q = float(val[(val.pool == "dentin") & (val.target == "youngs_modulus_GPa")
              & (val.split == "chemsys") & (val.features == "design")].conformal_q.iloc[0])
lo_m, hi_m = 10 ** -q, 10 ** q
en = pd.read_csv(os.path.join(DATA, "revision_enumeration_full.csv"))
dn = (en[en.pool == "dentin"].copy()          # formula breaks ties, so the order of
      .sort_values(["pred_modulus_GPa", "composition"])  # equal moduli is the same under
      .reset_index(drop=True))                   # every numpy sort implementation
dn["E_lo_GPa"] = (dn.pred_modulus_GPa * lo_m).round(2)
dn["E_hi_GPa"] = (dn.pred_modulus_GPa * hi_m).round(2)
dn["interval_contains_20"] = (dn.E_lo_GPa <= 20.0) & (20.0 <= dn.E_hi_GPa)
dn.head(500).to_csv(os.path.join(DATA, "audit_enumerated_dentin_lowest.csv"), index=False)
print(f"  dentin design-time interval: x[{lo_m:.3f}, {hi_m:.3f}]  (q = {q:.4f})")
print(f"  enumerated dentin formulas            : {len(dn):,}")
print(f"  lowest predicted modulus              : {dn.pred_modulus_GPa.min():.2f} GPa"
      f"  ({dn.composition.iloc[0]})")
print(f"  its interval                          : "
      f"[{dn.E_lo_GPa.iloc[0]:.1f}, {dn.E_hi_GPa.iloc[0]:.1f}] GPa")
print(f"  formulas whose interval contains 20 GPa: "
      f"{int(dn.interval_contains_20.sum()):,} of {len(dn):,} "
      f"({100.0 * dn.interval_contains_20.mean():.1f} percent)")
print(f"  formulas whose lower bound is at or below 20 GPa: "
      f"{int((dn.E_lo_GPa <= 20.0).sum()):,}")
print(f"  lowest lower bound anywhere in the space: {dn.E_lo_GPa.min():.1f} GPa")

print()
print("=" * 78)
print("E, F  validation on data that did not set the interval")
print("=" * 78)


load = load_pool


def design_predict(recs, tr, te, comp_cols, full_cols):
    """The released chain fitted on tr; log10 modulus of te with predicted structure."""
    c = Chain(recs, tr, comp_cols, full_cols, np.arange(len(tr)))
    return c.predict(te, "design")[0]


erows, frows = [], []
held = {}
for pool, fname, sfx in POOLS:
    df, comp_cols, full_cols = load(fname)
    recs = released(sfx)
    y = np.log10(df["youngs_modulus_GPa"].values)
    groups = df["chemsys"].values

    # E: nested grouped protocol. Each outer-test entry is predicted by the chain
    # fitted on the outer-training part, and its interval uses the quantile of the
    # inner out-of-fold residuals of that outer-training part alone, so no entry
    # whose coverage is assessed has contributed to its own quantile.
    pred = np.full(len(df), np.nan)
    qrow = np.full(len(df), np.nan)
    cov, widths = [], []
    for k, (tr, te) in enumerate(group_folds(groups, pool + "/chemsys", NFOLD)):
        inner = df.iloc[tr].reset_index(drop=True)
        gin = inner["chemsys"].values
        res = []
        for itr, ite in group_folds(gin, "%s/chemsys/outer%d" % (pool, k), NFOLD):
            p = design_predict(recs, inner.iloc[itr], inner.iloc[ite], comp_cols, full_cols)
            res.append(np.abs(np.log10(inner.iloc[ite]["youngs_modulus_GPa"].values) - p))
        res = np.concatenate(res)
        n = len(res)
        qq = np.sort(res)[min(int(np.ceil((n + 1) * COVERAGE)), n) - 1]
        pt = design_predict(recs, df.iloc[tr], df.iloc[te], comp_cols, full_cols)
        pred[te], qrow[te] = pt, qq
        hit = np.abs(y[te] - pt) <= qq
        cov.append(float(hit.mean()))
        widths.append(float(10 ** qq))
    covered = np.abs(y - pred) <= qrow
    held[pool] = (df, pred, qrow, covered)
    erows.append(dict(pool=pool, n=len(df), folds=NFOLD,
                      n_covered=int(covered.sum()),
                      nested_coverage=round(float(covered.mean()), 3),
                      nested_coverage_mean_of_folds=round(float(np.mean(cov)), 3),
                      nested_coverage_min_fold=round(float(np.min(cov)), 3),
                      mean_interval_multiplier=round(float(np.mean(widths)), 3),
                      nominal=COVERAGE))
    print(f"  {pool:14s} N={len(df):5d}  nested out-of-sample coverage "
          f"{int(covered.sum())}/{len(df)} = {covered.mean():.3f} (fold mean {np.mean(cov):.3f},"
          f" worst fold {np.min(cov):.3f}) at nominal {COVERAGE:.2f}, "
          f"mean multiplier x{np.mean(widths):.2f}")

    # F: low-modulus holdout
    for thr in (40.0, float(np.quantile(df["youngs_modulus_GPa"], 0.20))):
        te = df[df["youngs_modulus_GPa"] < thr]
        tr = df[df["youngs_modulus_GPa"] >= thr]
        if len(te) < 8:
            continue
        p = 10 ** design_predict(recs, tr, te, comp_cols, full_cols)
        t = te["youngs_modulus_GPa"].values
        frows.append(dict(pool=pool, threshold_GPa=round(thr, 1), n_held_out=len(te),
                          n_train=len(tr),
                          true_min=round(float(t.min()), 1),
                          true_median=round(float(np.median(t)), 1),
                          pred_min=round(float(p.min()), 1),
                          pred_median=round(float(np.median(p)), 1),
                          mae=round(float(mean_absolute_error(t, p)), 1),
                          mean_signed_error=round(float(np.mean(p - t)), 1),
                          r2=round(float(r2_score(t, p)), 3),
                          frac_pred_above_39_6=round(float(np.mean(p > 39.6)), 3),
                          frac_pred_above_20=round(float(np.mean(p > 20.0)), 3)))
        r = frows[-1]
        print(f"  {pool:14s} holdout below {thr:5.1f} GPa: n={len(te):4d}  "
              f"true median {r['true_median']:5.1f}  predicted median "
              f"{r['pred_median']:5.1f}  MAE {r['mae']:5.1f}  bias "
              f"{r['mean_signed_error']:+5.1f}  R2 {r['r2']:+.2f}  "
              f"predicted above 39.6 GPa: {100 * r['frac_pred_above_39_6']:.0f} percent")
    print(f"  {pool:14s} database moduli: min {df['youngs_modulus_GPa'].min():.1f}, "
          f"5th percentile {np.quantile(df['youngs_modulus_GPa'], 0.05):.1f}, "
          f"entries below 20 GPa: {int((df['youngs_modulus_GPa'] < 20).sum())}, "
          f"below 40 GPa: {int((df['youngs_modulus_GPa'] < 40).sum())}")

pd.DataFrame(erows).to_csv(os.path.join(DATA, "audit_nested_coverage.csv"), index=False)
pd.DataFrame(frows).to_csv(os.path.join(DATA, "audit_low_modulus_holdout.csv"), index=False)

print()
print("=" * 78)
print("G  the coefficient of determination in linear and log space")
print("=" * 78)
grows = []
for pool, fname, sfx in POOLS:
    df, comp_cols, full_cols = load(fname)
    yl = np.log10(df["youngs_modulus_GPa"].values)
    for kind in ("unshuffled", "shuffled", "chemsys"):
        if kind == "unshuffled":
            sp = KFold(NFOLD, shuffle=False).split(df)
        elif kind == "shuffled":
            sp = KFold(NFOLD, shuffle=True, random_state=SEED).split(df)
        else:
            sp = group_folds(df["chemsys"].values, pool + "/chemsys", NFOLD)
        oof = np.full(len(df), np.nan)
        per_fold = []
        for tr, te in sp:
            m = gbr().fit(df.iloc[tr][full_cols], yl[tr])
            p = m.predict(df.iloc[te][full_cols])
            oof[te] = p
            per_fold.append(r2_score(yl[te], p))
        lin_true, lin_pred = 10 ** yl, 10 ** oof
        grows.append(dict(pool=pool, protocol=kind, features="database",
                          r2_log_pooled=round(float(r2_score(yl, oof)), 3),
                          r2_log_mean_of_folds=round(float(np.mean(per_fold)), 3),
                          r2_linear_pooled=round(float(r2_score(lin_true, lin_pred)), 3),
                          mae_GPa=round(float(mean_absolute_error(lin_true, lin_pred)), 1)))
        r = grows[-1]
        print(f"  {pool:14s} {kind:11s} database descriptors: "
              f"R2(log, pooled) {r['r2_log_pooled']:.3f}  "
              f"R2(log, mean of folds) {r['r2_log_mean_of_folds']:.3f}  "
              f"R2(linear, pooled) {r['r2_linear_pooled']:.3f}  "
              f"MAE {r['mae_GPa']:.1f} GPa")
pd.DataFrame(grows).to_csv(os.path.join(DATA, "audit_r2_protocols.csv"), index=False)

print()
print("=" * 78)
print("H  what the deployed model predicts for the softest oxides in its own pool")
print("=" * 78)
# The holdout of F removes the whole low-modulus range from training, which a tree
# ensemble cannot extrapolate back. The fair test keeps those entries in the pool
# and holds out their chemistry, which is the protocol the design runs under.
hrows, srows = [], []
for pool, fname, sfx in POOLS:
    # the held-out predictions and training-only quantiles of the nested protocol (E)
    df, pred, qrow, covered = held[pool]
    df = df.copy()
    df["pred_E_GPa"] = 10 ** pred
    df["E_lo_GPa"] = df["pred_E_GPa"] * 10 ** -qrow
    df["E_hi_GPa"] = df["pred_E_GPa"] * 10 ** qrow
    df["covered"] = covered
    for cut in (20.0, 40.0):
        s_ = df[df["youngs_modulus_GPa"] < cut]
        if not len(s_):
            continue
        hrows.append(dict(pool=pool, cut_GPa=cut, n=len(s_),
                          n_covered=int(s_["covered"].sum()),
                          true_median=round(float(s_["youngs_modulus_GPa"].median()), 1),
                          pred_median=round(float(s_["pred_E_GPa"].median()), 1),
                          bias=round(float((s_["pred_E_GPa"] - s_["youngs_modulus_GPa"]).mean()), 1),
                          frac_pred_above_39_6=round(float((s_["pred_E_GPa"] > 39.6).mean()), 3),
                          frac_interval_covers_truth=round(
                              float(((s_["E_lo_GPa"] <= s_["youngs_modulus_GPa"])
                                     & (s_["youngs_modulus_GPa"] <= s_["E_hi_GPa"])).mean()), 3)))
        r = hrows[-1]
        print(f"  {pool:14s} measured below {cut:4.0f} GPa: n={len(s_):4d}  true median "
              f"{r['true_median']:5.1f}  predicted median {r['pred_median']:5.1f}  bias "
              f"{r['bias']:+6.1f}  predicted above 39.6 GPa "
              f"{100 * r['frac_pred_above_39_6']:5.1f} percent  interval covers the truth "
              f"{100 * r['frac_interval_covers_truth']:5.1f} percent")
    df[["material_id", "formula", "youngs_modulus_GPa", "pred_E_GPa",
        "E_lo_GPa", "E_hi_GPa", "covered"]].assign(pool=pool).to_csv(
        os.path.join(DATA, "audit_oof_%s.csv" % pool), index=False)
    low = df.nsmallest(10, "youngs_modulus_GPa")
    for _, r in low.iterrows():
        srows.append(dict(pool=pool, formula=r["formula"],
                          material_id=r["material_id"],
                          measured_E_GPa=round(float(r["youngs_modulus_GPa"]), 1),
                          pred_E_GPa=round(float(r["pred_E_GPa"]), 1),
                          E_lo_GPa=round(float(r["E_lo_GPa"]), 1),
                          E_hi_GPa=round(float(r["E_hi_GPa"]), 1)))
    print(f"  {pool:14s} ten softest entries, measured against out-of-fold prediction:")
    for r in srows[-10:]:
        print(f"      {r['formula']:18s} {r['measured_E_GPa']:6.1f}  ->  "
              f"{r['pred_E_GPa']:6.1f}  [{r['E_lo_GPa']:5.1f}, {r['E_hi_GPa']:6.1f}]")
pd.DataFrame(hrows).to_csv(os.path.join(DATA, "audit_low_modulus_recall.csv"), index=False)
pd.DataFrame(srows).to_csv(os.path.join(DATA, "audit_softest_entries.csv"), index=False)

print()
print("REVISION_AUDIT_V2_DONE", flush=True)
