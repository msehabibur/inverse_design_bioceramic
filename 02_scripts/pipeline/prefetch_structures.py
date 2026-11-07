"""
Login-node prefetch for the MLFF cross-check (needs MP network; compute nodes
have none). Two products, both saved to scratch:

1. A modulus-spanning sample of REAL MP oxides (that have DFT elasticity) with
   their structures -> lets the compute node compute CHGNet moduli and show a
   CHGNet-vs-DFT parity (validates the MLFF as a structure-aware modulus tool,
   with NO DFT run by us).

2. For each of the 45 GA-designed candidates: the nearest real MP compound in
   its chemical system, with that compound's DFT energy-above-hull and (if
   available) DFT modulus -- a synthesizable, stability-known anchor for each
   design.

Outputs: data/mlff_sample_structures.json, data/designed_nearest_analog.csv
"""
import os, json
import numpy as np
import pandas as pd
from pymatgen.core import Composition
from mp_api.client import MPRester
import os as _os, sys as _sys; _sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))  # noqa: make paths/features importable
from paths import DATA

KEY = os.environ.get("MP_API_KEY") or open(os.path.join(DATA, "mp_key.txt")).read().strip()
N_SAMPLE = 18
MAX_SITES = 20

# ---- 1. modulus-spanning sample of real MP oxides ------------------------
mp = pd.read_csv(os.path.join(DATA, "mp_data.csv")).dropna(subset=["youngs_modulus_GPa"])
mp = mp.sort_values("youngs_modulus_GPa").reset_index(drop=True)
idx = np.linspace(0, len(mp) - 1, N_SAMPLE * 2).astype(int)   # oversample; filter by size
sample = mp.iloc[idx].drop_duplicates("material_id")

structs, kept = [], 0
with MPRester(KEY) as mpr:
    for r in sample.itertuples(index=False):
        if kept >= N_SAMPLE:
            break
        try:
            s = mpr.get_structure_by_material_id(r.material_id)
        except Exception as e:
            print("skip", r.material_id, e); continue
        if len(s) > MAX_SITES:
            continue
        structs.append({"material_id": str(r.material_id), "formula": r.formula,
                        "dft_E_GPa": float(r.youngs_modulus_GPa),
                        "structure": s.as_dict()})
        kept += 1
        print(f"  fetched {r.material_id} {r.formula}  DFT_E={r.youngs_modulus_GPa:.0f}  sites={len(s)}")
json.dump(structs, open(os.path.join(DATA, "mlff_sample_structures.json"), "w"))
print(f"saved {len(structs)} structures -> mlff_sample_structures.json")

# ---- 2. nearest real MP compound for each designed candidate -------------
def frac_vec(comp, elements):
    f = comp.fractional_composition
    return np.array([f[e] if e in f else 0.0 for e in elements])

SHOWCASES = [("enamel", "designed_compositions.csv"),
             ("dentin", "designed_dentin_compositions.csv"),
             ("implant", "designed_implant_compositions.csv")]
rows = []
with MPRester(KEY) as mpr:
    for sc, csv_file in SHOWCASES:
        df = pd.read_csv(os.path.join(DATA, csv_file))
        for rank, comp_str in enumerate(df["composition"], 1):
            comp = Composition(comp_str)
            els = sorted([str(e) for e in comp.elements])
            try:
                docs = mpr.materials.summary.search(
                    chemsys="-".join(els),
                    fields=["formula_pretty", "composition", "energy_above_hull"])
            except Exception as e:
                docs = []
            best, bestd = None, 1e9
            tv_els = els
            tv = frac_vec(comp, tv_els)
            for d in docs:
                dc = Composition(d.composition)
                dist = np.linalg.norm(frac_vec(dc, tv_els) - tv)
                if dist < bestd:
                    bestd, best = dist, d
            rows.append({
                "showcase": sc, "rank": rank, "designed": comp.reduced_formula,
                "nearest_mp": best.formula_pretty if best else "",
                "nearest_dist": round(float(bestd), 3) if best else None,
                "nearest_e_above_hull_eV": round(float(best.energy_above_hull), 3)
                    if best and best.energy_above_hull is not None else None,
            })
            print(f"  [{sc}] {comp.reduced_formula} -> nearest {rows[-1]['nearest_mp']} "
                  f"(d={rows[-1]['nearest_dist']}, Ehull={rows[-1]['nearest_e_above_hull_eV']})")
pd.DataFrame(rows).to_csv(os.path.join(DATA, "designed_nearest_analog.csv"), index=False)
print("saved -> designed_nearest_analog.csv")
