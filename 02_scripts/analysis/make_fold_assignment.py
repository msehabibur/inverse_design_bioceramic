"""Write the grouped fold assignment that every validation script reads.

GroupKFold deals the groups to folds in order of size and breaks ties between groups of
equal size with numpy's unstable argsort. The grouped folds of the published results
were drawn under numpy 2.4.6 with scikit-learn 1.6.1; numpy 1.26.4 orders the equally
sized chemical systems differently and so gives other folds. This
script records, for every pool, the fold of every reduced formula and of every chemical
system, and for the nested coverage test the inner fold of every chemical system of each
outer-training part. chain.group_folds reads the file back, so the folds no longer depend
on the numpy version.

Run once, under the environment that produced the published results:

    05_config_envs/site_venv/bin/python 02_scripts/analysis/make_fold_assignment.py

Writes 01_data/fold_assignment.csv (key, group, fold).
"""
import os
import sys

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chain import FOLDS, POOLS, load_pool

NFOLD = 5
rows = []


def record(key, groups):
    for k, (_, te) in enumerate(GroupKFold(NFOLD).split(groups, groups=groups)):
        for g in sorted(set(groups[te])):
            rows.append((key, g, k))


print("numpy %s, scikit-learn %s" % (np.__version__, sklearn.__version__), flush=True)
for pool, fname, _ in POOLS:
    df, _, _ = load_pool(fname)
    record(pool + "/formula", df["reduced_formula"].values)
    groups = df["chemsys"].values
    record(pool + "/chemsys", groups)
    for k, (tr, _) in enumerate(GroupKFold(NFOLD).split(df, groups=groups)):
        record("%s/chemsys/outer%d" % (pool, k), groups[tr])
    print("  %s: %d formulas, %d chemical systems" % (
        pool, df["reduced_formula"].nunique(), len(set(groups))), flush=True)

pd.DataFrame(rows, columns=["key", "group", "fold"]).to_csv(FOLDS, index=False)
print("wrote %s, %d rows" % (FOLDS, len(rows)), flush=True)
print("FOLD_ASSIGNMENT_DONE", flush=True)
