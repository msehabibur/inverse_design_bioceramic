#!/usr/bin/env python3
"""Identify a force-field checkpoint by re-relaxing recorded competitor phases.

The run records name the CHGNet, MACE and MatterSim checkpoints but not the M3GNet and
SevenNet ones. This script relaxes 22 converged competitor phases with one candidate
checkpoint under the settings of relax_worker.py (FIRE with a Frechet cell filter,
fmax 0.05 eV/A) and compares each energy per atom with the value in results/.
The checkpoint that was used reproduces the recorded energies to about 1 meV/atom.

Usage: checkpoint_check.py --model {m3gnet,omni,chgnet} --checkpoint NAME [--modal M]
       [--revision HF_REVISION] --out OUT.csv
  m3gnet  NAME is a matgl model name or a Hugging Face repo; --revision pins its commit
  omni    NAME is a SevenNet checkpoint keyword, --modal its fidelity head
  chgnet  NAME is ignored; CHGNet.load() gives the installed default
"""
import os, sys, json, csv, glob, argparse, warnings
warnings.filterwarnings("ignore")
import numpy as np
from ase import Atoms
from ase.optimize import FIRE
from ase.filters import FrechetCellFilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import ROOT
BASE = os.path.join(ROOT, "06_mlff_stability")

IDS = ["comp_mpaaaaaabt", "comp_mpaaaaaafe", "comp_mpaaaaabwr", "comp_mpaaaaadwf",
       "comp_mpaaaaabmg", "comp_mpaaaaapvc", "comp_mpaaaaadns", "comp_mpaaaaaaft",
       "comp_mpaaaaachq", "comp_mpaaaaactu", "comp_mpaaaaaacu", "comp_mpaaaaablj",
       "comp_mpaaaaadmm", "comp_mpaaaabdws", "comp_mpaaaaacid", "comp_mpaaacqrmy",
       "comp_mpaaabfcwo", "comp_mpaaaaaeyv", "comp_mpaaaaagwe", "comp_mpaaaaahpk",
       "comp_mpaaaaalpw", "comp_mpaaaaagwx"]


def load_atoms(sid):
    d = json.load(open(os.path.join(BASE, "structures", sid + ".json")))
    return Atoms(symbols=[s["species"][0]["element"] for s in d["sites"]],
                 scaled_positions=[s["abc"] for s in d["sites"]],
                 cell=d["lattice"]["matrix"], pbc=True)


def calculator(a):
    if a.model == "m3gnet":
        import matgl
        from matgl.ext.ase import PESCalculator
        name = a.checkpoint
        if a.revision:
            from huggingface_hub import snapshot_download
            name = snapshot_download(a.checkpoint, revision=a.revision)
        return PESCalculator(matgl.load_model(name), stress_unit="eV/A3")  # ASE cell filters expect eV/A3
    if a.model == "omni":
        from sevenn.calculator import SevenNetCalculator
        kw = {"modal": a.modal} if a.modal else {}
        return SevenNetCalculator(a.checkpoint, device="cpu", **kw)
    from chgnet.model import CHGNet, CHGNetCalculator
    return CHGNetCalculator(CHGNet.load(), use_device="cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["m3gnet", "omni", "chgnet"])
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--modal", default="")
    ap.add_argument("--revision", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    rec = {}
    for f in glob.glob(os.path.join(BASE, "results", f"{a.model}.c*.csv")):
        for r in csv.DictReader(open(f)):
            if r["energy_per_atom"] != "nan":
                rec[r["sid"]] = r
    calc = calculator(a)
    label = a.checkpoint + (f"@{a.revision[:7]}" if a.revision else "") + \
        (f" ({a.modal})" if a.modal else "")
    diffs = []
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sid", "composition", "model", "checkpoint", "recorded_eV_per_atom",
                    "rerun_eV_per_atom", "difference_meV_per_atom"])
        for sid in IDS:
            r = rec[sid]
            atoms = load_atoms(sid)
            atoms.calc = calc
            FIRE(FrechetCellFilter(atoms), logfile=None).run(fmax=0.05, steps=400)
            e = atoms.get_potential_energy() / len(atoms)
            d = 1000 * (e - float(r["energy_per_atom"]))
            diffs.append(d)
            w.writerow([sid, r["composition"], a.model, label, r["energy_per_atom"],
                        f"{e:.6f}", f"{d:.3f}"])
    d = np.abs(diffs)
    print(f"{label}: mean |difference| {d.mean():.3f} meV/atom, largest {d.max():.3f} "
          f"meV/atom over {len(d)} phases")


if __name__ == "__main__":
    main()
