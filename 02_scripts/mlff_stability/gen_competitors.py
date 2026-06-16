#!/usr/bin/env python3
"""Phase 1b (login node): fetch MP-stable competing phases per chemsys; merge manifest."""
import os, sys, json, time, importlib
import pandas as pd
from pymatgen.core import Composition
from pymatgen.analysis.phase_diagram import PhaseDiagram
from mp_api.client import MPRester
for o, n in {"pymatgen.core.entries": "pymatgen.entries.computed_entries",
             "pymatgen.analysis.compatibility": "pymatgen.entries.compatibility"}.items():
    try: sys.modules[o] = importlib.import_module(n)
    except Exception: pass

BASE = "/anvil/scratch/x-mrahman2/Purdue_Projects/Inverse_Design_of_Bioceramics_by_Machine_Learning/"
DATA = BASE + "01_data/"
OUT = BASE + "06_mlff_stability/"
STR = OUT + "structures/"
KEY = open(DATA + "mp_key.txt").read().strip()

cand_man = pd.read_csv(OUT + "candidates_manifest.csv")
chemsystems = sorted(cand_man["chemsys"].unique())
mpr = MPRester(KEY)
man, seen, chemsys_map = [], set(), {}
for cs in chemsystems:
    els = cs.split("-")
    t0 = time.time()
    ents = mpr.get_entries_in_chemsys(els)
    stable = PhaseDiagram(ents).stable_entries
    sids = []
    for e in stable:
        mpid = str(e.data.get("material_id", e.entry_id))
        sid = "comp_" + mpid.replace("mp-", "mp")
        if mpid not in seen:
            seen.add(mpid)
            json.dump(e.structure.as_dict(), open(STR + sid + ".json", "w"))
            man.append(dict(sid=sid, kind="competitor", showcase="", rank="",
                            composition=e.composition.formula,
                            reduced_formula=e.composition.reduced_formula,
                            chemsys=e.composition.chemical_system, mp_id=mpid,
                            seed="", nsites=len(e.structure)))
        sids.append(sid)
    chemsys_map[cs] = sids
    print(f"{cs:18s} stable={len(stable)} total_unique={len(seen)} ({time.time()-t0:.1f}s)", flush=True)

comp_man = pd.DataFrame(man)
comp_man.to_csv(OUT + "competitors_manifest.csv", index=False)
json.dump(chemsys_map, open(OUT + "chemsys_map.json", "w"))
full = pd.concat([cand_man, comp_man], ignore_index=True)
full.to_csv(OUT + "manifest.csv", index=False)
print(f"DONE competitors_unique={len(comp_man)} manifest_total={len(full)}")
