#!/usr/bin/env python3
"""Phase 3 (login, bioceramics_venv): build per-model MLFF convex hulls from relaxed
competitor energies, place each candidate (min-energy seed), compute energy-above-hull
+ decomposition, and report cross-model consensus. -> stability_mlff.csv + summary."""
import os, sys, json, glob, importlib
import numpy as np, pandas as pd
from pymatgen.core import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
for o, n in {"pymatgen.core.entries": "pymatgen.entries.computed_entries",
             "pymatgen.analysis.compatibility": "pymatgen.entries.compatibility"}.items():
    try: sys.modules[o] = importlib.import_module(n)
    except Exception: pass

BASE = "/anvil/scratch/x-mrahman2/Purdue_Projects/Inverse_Design_of_Bioceramics_by_Machine_Learning/06_mlff_stability/"
MODELS = ["chgnet", "m3gnet", "mace", "mattersim", "grace", "omni"]

man = pd.read_csv(BASE + "manifest.csv")
chemsys_map = json.load(open(BASE + "chemsys_map.json"))
man_by_sid = {r.sid: r for r in man.itertuples()}

def load_model(model):
    files = sorted(glob.glob(BASE + f"results/{model}.c*.csv")) + \
            ([BASE + f"results/{model}.csv"] if os.path.exists(BASE + f"results/{model}.csv") else [])
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    df = df.dropna(subset=["energy_eV"])
    df = df[~df["energy_eV"].astype(str).str.contains("nan")]
    df["energy_eV"] = df["energy_eV"].astype(float)
    df = df.sort_values("sid").drop_duplicates("sid", keep="last")
    return df.set_index("sid")

candidates = man[man.kind == "candidate"][["showcase", "rank", "composition", "chemsys"]].drop_duplicates()
candidates = candidates.sort_values(["showcase", "rank"])

rows = []
per_model_decomp = {}
for model in MODELS:
    df = load_model(model)
    if df is None:
        print(f"[skip] {model}: no results yet"); continue
    emap = df["energy_eV"].to_dict()           # sid -> total energy (eV)
    # competitor PDEntries
    comp_entries = {}                          # sid -> PDEntry
    for sid in man[man.kind == "competitor"].sid:
        if sid in emap:
            comp = Composition(man_by_sid[sid].composition)
            comp_entries[sid] = PDEntry(comp, emap[sid])
    print(f"[{model}] competitors with energy: {len(comp_entries)}")
    # phase diagram per chemsys (cache)
    pds = {}
    for cs, sids in chemsys_map.items():
        ents = [comp_entries[s] for s in sids if s in comp_entries]
        try:
            pds[cs] = PhaseDiagram(ents)
        except Exception as e:
            pds[cs] = None
            print(f"   PD fail {cs}: {str(e)[:60]}")
    # candidates
    for c in candidates.itertuples():
        seeds = man[(man.kind == "candidate") & (man.showcase == c.showcase) & (man["rank"] == c.rank)]
        epa = [(s, emap[s] / man_by_sid[s].nsites) for s in seeds.sid if s in emap]
        key = f"{c.showcase}_{c.rank}"
        if not epa or pds.get(c.chemsys) is None:
            rows.append(dict(showcase=c.showcase, rank=c.rank, composition=c.composition,
                             chemsys=c.chemsys, model=model, ehull=np.nan, ef_hull=np.nan, nseeds=len(epa)))
            continue
        best_sid, best_epa = min(epa, key=lambda t: t[1])
        comp = Composition(c.composition)
        cand_entry = PDEntry(comp, best_epa * comp.num_atoms)
        pdg = pds[c.chemsys]
        try:
            decomp, ehull = pdg.get_decomp_and_e_above_hull(cand_entry, allow_negative=True)
            dec_str = " + ".join(f"{e.composition.reduced_formula}({a:.2f})"
                                  for e, a in sorted(decomp.items(), key=lambda kv: -kv[1]))
            if ehull > 1e-6:
                per_model_decomp.setdefault(key, []).append(dec_str)
        except Exception as e:
            try:
                ehull = pdg.get_e_above_hull(cand_entry)
            except Exception:
                ehull = np.nan
            dec_str = f"ERR {str(e)[:40]}"
        try:
            ef = pdg.get_form_energy_per_atom(cand_entry)
        except Exception:
            ef = np.nan
        rows.append(dict(showcase=c.showcase, rank=c.rank, composition=c.composition,
                         chemsys=c.chemsys, model=model, ehull=ehull, ef_hull=ef,
                         nseeds=len(epa), decomposition=dec_str))

long = pd.DataFrame(rows)
long.to_csv(BASE + "stability_mlff_long_raw.csv", index=False)
# sanity filter: drop physically implausible per-model ehull (broken hull from a
# diverged competitor relaxation). True above-hull values are O(0-2) eV/atom.
n_bad = int((long["ehull"].abs() > 3).sum())
long.loc[long["ehull"].abs() > 3, "ehull"] = np.nan
print(f"[sanity] dropped {n_bad} per-model ehull values with |ehull|>3 eV/atom (broken hulls)")
long.to_csv(BASE + "stability_mlff_long.csv", index=False)

# consensus wide table
piv = long.pivot_table(index=["showcase", "rank", "composition", "chemsys"],
                       columns="model", values="ehull")
piv["ehull_median"] = piv[[m for m in MODELS if m in piv.columns]].median(axis=1)
piv["ehull_mean"] = piv[[m for m in MODELS if m in piv.columns]].mean(axis=1)
piv["ehull_min"] = piv[[m for m in MODELS if m in piv.columns]].min(axis=1)
piv["ehull_max"] = piv[[m for m in MODELS if m in piv.columns]].max(axis=1)
piv["n_models"] = piv[[m for m in MODELS if m in piv.columns]].notna().sum(axis=1)
piv = piv.reset_index()
# modal decomposition across models
def modal(key):
    ds = per_model_decomp.get(key, [])
    return max(set(ds), key=ds.count) if ds else ""
piv["decomposition_consensus"] = [modal(f"{r.showcase}_{r.rank}") for r in piv.itertuples()]
piv.to_csv(BASE + "stability_mlff.csv", index=False)

print("\n=== consensus energy above hull (eV/atom) ===")
cols = ["showcase", "rank", "composition", "ehull_median", "ehull_min", "ehull_max", "n_models"]
print(piv[cols].to_string(index=False))
print(f"\nmean median-ehull: {piv['ehull_median'].mean():.3f} eV/atom")
print(f"candidates with median ehull > 0.05: {(piv['ehull_median'] > 0.05).sum()}/{len(piv)}")
print(f"candidates with median ehull <= 0.025 (near-stable): {(piv['ehull_median'] <= 0.025).sum()}/{len(piv)}")
