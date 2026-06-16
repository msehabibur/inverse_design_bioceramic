#!/usr/bin/env python3
"""Phase 2 (GPU, per-model env): relax structures with one MLFF and record energies.
Pure JSON->ASE loader so it needs only ASE + the model package (no pymatgen).
Usage: relax_worker.py --model MODEL --manifest M.csv --out OUT.csv
       [--kinds candidate,competitor] [--ids id1,id2] [--steps 300] [--fmax 0.05]
       [--fixcell]"""
import os, sys, json, csv, time, argparse, warnings, signal
warnings.filterwarnings("ignore")
os.environ.setdefault("PYTHONWARNINGS", "ignore")
import numpy as np

class _Timeout(Exception):
    pass

def _alarm_handler(signum, frame):
    raise _Timeout()
ZOO = "/anvil/scratch/x-mrahman2/Purdue_Projects/Chalcogenide_Defect_Screening/05_config_envs/mlff_zoo"
sys.path.insert(0, ZOO)
BASE = "/anvil/scratch/x-mrahman2/Purdue_Projects/Inverse_Design_of_Bioceramics_by_Machine_Learning/06_mlff_stability/"
STR = BASE + "structures/"

from ase import Atoms
from ase.optimize import FIRE
try:
    from ase.filters import FrechetCellFilter as CellFilter
except Exception:
    from ase.constraints import ExpCellFilter as CellFilter

def load_atoms(sid):
    d = json.load(open(STR + sid + ".json"))
    cell = d["lattice"]["matrix"]
    symbols, scaled = [], []
    for s in d["sites"]:
        sp = s["species"][0]["element"]
        symbols.append(sp)
        scaled.append(s["abc"])
    return Atoms(symbols=symbols, scaled_positions=scaled, cell=cell, pbc=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kinds", default="candidate,competitor")
    ap.add_argument("--ids", default="")
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--fmax", type=float, default=0.05)
    ap.add_argument("--fixcell", action="store_true")
    ap.add_argument("--chunk", type=int, default=0)
    ap.add_argument("--nchunks", type=int, default=1)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--timeout", type=int, default=0)   # per-structure wall seconds; 0=off
    a = ap.parse_args()
    if a.timeout > 0:
        signal.signal(signal.SIGALRM, _alarm_handler)

    if a.device == "cpu":
        nthreads = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
        os.environ.setdefault("OMP_NUM_THREADS", str(nthreads))
        try:
            import torch
            torch.set_num_threads(nthreads)
        except Exception:
            pass

    from mlff13_calculators import get_calculator
    calc = get_calculator(a.model, device=a.device)

    rows = list(csv.DictReader(open(a.manifest)))
    kinds = set(a.kinds.split(","))
    idset = set(x for x in a.ids.split(",") if x)
    todo = [r for r in rows if r["kind"] in kinds and (not idset or r["sid"] in idset)]
    # deterministic chunking for SLURM arrays
    todo = [r for j, r in enumerate(todo) if j % a.nchunks == a.chunk]

    done = set()
    if os.path.exists(a.out):
        for r in csv.DictReader(open(a.out)):
            done.add(r["sid"])
    f = open(a.out, "a", newline="")
    w = csv.writer(f)
    if not done:
        w.writerow(["sid", "model", "kind", "composition", "energy_eV",
                    "energy_per_atom", "nsites", "nsteps", "converged"])
        f.flush()

    print(f"MODEL {a.model}: {len(todo)} structures ({len(done)} already done)", flush=True)
    for i, r in enumerate(todo):
        sid = r["sid"]
        if sid in done:
            continue
        t0 = time.time()
        if a.timeout > 0:
            signal.alarm(a.timeout)
        try:
            atoms = load_atoms(sid)
            atoms.calc = calc
            n = len(atoms)
            opt_atoms = atoms if a.fixcell else CellFilter(atoms)
            dyn = FIRE(opt_atoms, logfile=None)
            dyn.run(fmax=a.fmax, steps=a.steps)
            e = float(atoms.get_potential_energy())
            conv = bool(np.max(np.linalg.norm(atoms.get_forces(), axis=1)) < a.fmax)
            w.writerow([sid, a.model, r["kind"], r.get("composition", ""),
                        f"{e:.6f}", f"{e/n:.6f}", n, dyn.get_number_of_steps(), int(conv)])
            f.flush()
            print(f"  [{i+1}/{len(todo)}] {sid:30s} E/atom={e/n:.4f} n={n} "
                  f"steps={dyn.get_number_of_steps()} conv={conv} ({time.time()-t0:.1f}s)", flush=True)
        except Exception as ex:
            w.writerow([sid, a.model, r["kind"], r.get("composition", ""), "nan", "nan",
                        "", "", "0"])
            f.flush()
            tag = "TIMEOUT" if isinstance(ex, _Timeout) else f"FAIL {type(ex).__name__}"
            print(f"  [{i+1}/{len(todo)}] {sid:30s} {tag}: {str(ex)[:60]}", flush=True)
        finally:
            if a.timeout > 0:
                signal.alarm(0)
    f.close()
    print("WORKER_DONE", flush=True)

if __name__ == "__main__":
    main()
