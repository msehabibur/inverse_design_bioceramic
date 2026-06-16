#!/usr/bin/env python3
"""Phase 1a (SLURM CPU node): pyxtal random symmetric crystals for the 45 candidates."""
import os, sys, json, random
import pandas as pd
from pymatgen.core import Composition
from pyxtal import pyxtal

random.seed(12345)
BASE = "/anvil/scratch/x-mrahman2/Purdue_Projects/Inverse_Design_of_Bioceramics_by_Machine_Learning/"
DATA = BASE + "01_data/"
OUT = BASE + "06_mlff_stability/"
STR = OUT + "structures/"
os.makedirs(STR, exist_ok=True)
K = 8
files = {"enamel": "designed_compositions.csv",
         "dentin": "designed_dentin_compositions.csv",
         "implant": "designed_implant_compositions.csv"}
man = []
for sc, fn in files.items():
    df = pd.read_csv(DATA + fn)
    for i, row in df.iterrows():
        c = Composition(row["composition"])
        species = [str(e) for e in c.elements]
        counts = [int(round(c[e])) for e in c.elements]
        groups = list(range(2, 231)); random.shuffle(groups)
        ok = 0
        for g in groups:
            if ok >= K: break
            x = pyxtal()
            try:
                x.from_random(3, g, species, counts)
                if not x.valid: continue
                s = x.to_pymatgen()
            except Exception:
                continue
            sid = f"cand_{sc}_{int(i)+1:02d}_s{ok}"
            json.dump(s.as_dict(), open(STR + sid + ".json", "w"))
            man.append(dict(sid=sid, kind="candidate", showcase=sc, rank=int(i)+1,
                            composition=row["composition"], reduced_formula=c.reduced_formula,
                            chemsys=c.chemical_system, mp_id="", seed=g, nsites=len(s)))
            ok += 1
        print(f"{sc:7s} {int(i)+1:2d} {row['composition']:22s} {ok}/{K}", flush=True)
pd.DataFrame(man).to_csv(OUT + "candidates_manifest.csv", index=False)
print(f"DONE candidates={len(man)}")
