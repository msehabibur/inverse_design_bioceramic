#!/usr/bin/env python3
"""Rebuild Table S_mlff with 5 models (GRACE dropped) and replace it in main.tex."""
import re, math
import pandas as pd
B = "/anvil/scratch/x-mrahman2/Purdue_Projects/Inverse_Design_of_Bioceramics_by_Machine_Learning/"
M = B + "04_manuscript/main.tex"
df = pd.read_csv(B + "06_mlff_stability/stability_mlff.csv")

def subf(f):
    return re.sub(r'([A-Za-z\)])(\d+)', r'\1$_{\2}$', f.strip())

def fmt_decomp(s):
    if not isinstance(s, str) or not s.strip():
        return "--"
    out = []
    for t in s.split(" + "):
        t = t.strip()
        if t.endswith(")") and "(" in t:
            formula, frac = t[:-1].rsplit("(", 1)
            out.append(f"{subf(formula)} ({frac})")
        else:
            out.append(subf(t))
    return " + ".join(out)

def num(x):
    return "--" if (x is None or (isinstance(x, float) and math.isnan(x))) else f"{x:.2f}"

CASE = {"enamel": "Enamel", "dentin": "Dentin", "implant": "Implant"}
order = {"enamel": 0, "dentin": 1, "implant": 2}
df = df.sort_values(by=["showcase", "rank"], key=lambda c: c.map(order) if c.name == "showcase" else c)

rows = []
for r in df.itertuples():
    cells = [CASE[r.showcase], subf(r.composition),
             num(r.chgnet), num(r.m3gnet), num(r.mace), num(r.mattersim),
             num(r.omni), num(r.ehull_median), fmt_decomp(r.decomposition_consensus)]
    rows.append(" & ".join(cells) + r" \\")

hdr = (r"Case & Composition & CHGNet & M3GNet & MACE & MatterSim & SevenNet & "
       r"Median & Decomposition (consensus) \\")
tbl = ["{\\scriptsize",
       r"\begin{longtable}{l l r r r r r r p{0.32\textwidth}}",
       r"\caption{\textbf{Five-model machine-learning force-field energy above hull for all 45 genetic-algorithm candidates.} "
       r"Per-model and consensus (median) energy above hull $E_{\rm hull}$ in eV/atom (CHGNet, M3GNet, MACE, MatterSim, SevenNet), "
       r"and the consensus equilibrium decomposition (mole fractions in parentheses). A dash denotes a model whose hull could not be "
       r"formed for that candidate. All values are upper bounds (random-structure relaxation). GRACE was evaluated but excluded "
       r"(it returned unphysical sub-hull energies for these random structures).}\label{tab:Smlff}\\",
       r"\toprule " + hdr + r" \midrule",
       r"\endfirsthead \toprule " + hdr + r" \midrule \endhead"]
tbl.extend(rows)
tbl.append(r"\bottomrule")
tbl.append(r"\end{longtable}}")
block = "\n".join(tbl) + "\n"

s = open(M).read()
# replace existing tab:Smlff block: from the "{\scriptsize" before it to its "\end{longtable}}"
a = s.index("{\\scriptsize")
b = s.index("\\end{longtable}}", a) + len("\\end{longtable}}")
old = s[a:b]
assert "tab:Smlff" in old, "tab:Smlff not in located block"
s2 = s[:a] + block.rstrip("\n") + s[b:]
open(M, "w").write(s2)
print("replaced tab:Smlff (5 models); old %d -> new %d chars" % (len(old), len(block)))
