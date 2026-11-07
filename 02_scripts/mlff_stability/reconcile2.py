import glob, csv, os, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ROOT
B = os.path.join(ROOT, "06_mlff_stability") + "/"
MODELS=["chgnet","m3gnet","mace","mattersim","grace","omni"]
man=pd.read_csv(B+"manifest.csv")
comp=man[man.kind=="competitor"]["sid"].tolist()
cands=man[man.kind=="candidate"]
cand_groups=cands.groupby(["showcase","rank"])["sid"].apply(list)
for m in MODELS:
    valid=set()
    for f in glob.glob(B+f"results/{m}.c*.csv"):
        for r in csv.DictReader(open(f)):
            e=str(r.get("energy_eV","nan"))
            if r.get("sid") and e not in ("nan","","NaN"): valid.add(r["sid"])
    miss_comp=[s for s in comp if s not in valid]
    zero_seed=sum(1 for _,seeds in cand_groups.items() if not any(s in valid for s in seeds))
    open(B+f"compmiss_{m}.txt","w").write(",".join(miss_comp))
    print(f"{m:10s} valid={len(valid):4d}  MISSING_COMPETITORS={len(miss_comp):3d}  candidates_with_0_seeds={zero_seed}/45")
