"""
The released model chain, rebuilt for validation.

Every validation, coverage and parity analysis takes its estimators from here, so the
models it scores are the ones the genetic algorithm used: the estimator class and every
hyperparameter are read from the released 01_data/model_*.joblib files and cloned
unfitted, then fitted inside each fold on the rows train.py / train_dentin.py would have
used (modulus and volume per atom on the whole training part, hardness on its rows with
a positive Chen hardness).

Released chain (see train.py): enamel and implant pool = gradient-boosting modulus,
random-forest hardness, gradient-boosting volume per atom; dentin pool = gradient-boosting
modulus, random-forest hardness, random-forest volume per atom.

Run under scikit-learn 1.6.1 (05_config_envs/requirements.txt), the version the models
were pickled with. check_released() refits each clone on the full pool and requires it to
reproduce the released predictions, which tests the features, the rows and the estimator
settings at once.
"""
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from pymatgen.core import Composition

from features import make_featurizer
from paths import DATA

AMU_PER_A3_TO_GCC = 1.66054                     # matches inverse_ga*.py
POOLS = (("enamel_implant", "mp_data.csv", ""), ("dentin", "dentin_data.csv", "_dentin"))
FILES = {"E": "model_modulus%s.joblib", "H": "model_hardness%s.joblib",
         "V": "model_volperatom%s.joblib"}


def load_pool(fname):
    """The training table exactly as train.py builds it, plus grouping columns."""
    df = pd.read_csv(os.path.join(DATA, fname))
    df["composition_obj"] = df["formula"].apply(Composition)
    f = make_featurizer()
    df = f.featurize_dataframe(df, col_id="composition_obj", ignore_errors=True)
    comp_cols = f.feature_labels()
    full_cols = comp_cols + ["density", "vol_per_atom"]
    df = df.dropna(subset=full_cols).reset_index(drop=True)
    k = df["G_GPa"] / df["K_GPa"]
    df["hardness_GPa"] = 2 * (k ** 2 * df["G_GPa"]) ** 0.585 - 3
    df["mean_mass"] = [c.weight / c.num_atoms for c in df["composition_obj"]]
    df["reduced_formula"] = [c.reduced_formula for c in df["composition_obj"]]
    df["chemsys"] = [c.chemical_system for c in df["composition_obj"]]
    return df, comp_cols, full_cols


def released(suffix):
    """The three released model records of one pool, keyed E, H, V."""
    recs = {k: joblib.load(os.path.join(DATA, v % suffix)) for k, v in FILES.items()}
    for r in recs.values():                     # the forests were saved with n_jobs=-1,
        if "n_jobs" in r["model"].get_params():  # which exceeds the login-node thread cap
            r["model"].set_params(n_jobs=4)
    return recs


FOLDS = os.path.join(DATA, "fold_assignment.csv")


def group_folds(groups, key, nfold=5):
    """The released grouped folds for one pool and grouping, as (train, test) indices.

    GroupKFold deals the groups to folds in order of size and breaks ties between groups
    of equal size with numpy's unstable argsort, so the same data give different folds
    under different numpy versions. The assignment behind the published results is
    therefore released in 01_data/fold_assignment.csv (make_fold_assignment.py) and read
    here. key names the pool and the grouping, and for the inner folds of the nested
    coverage test also the outer fold, e.g. "dentin/chemsys/outer2". The indices are
    the ones GroupKFold(5).split returned when the file was written.
    """
    t = pd.read_csv(FOLDS, keep_default_na=False)
    t = t[t["key"] == key]
    if t.empty:
        raise SystemExit("no fold assignment for %s in %s" % (key, FOLDS))
    f = pd.Series(t["fold"].values, index=t["group"].values).reindex(groups).values
    if pd.isna(f).any():
        raise SystemExit("%s: %d entries belong to no released fold"
                         % (key, int(pd.isna(f).sum())))
    return [(np.where(f != k)[0], np.where(f == k)[0]) for k in range(nfold)]


def fresh(rec):
    """An unfitted copy of a released estimator, same class and hyperparameters."""
    m = clone(rec["model"])
    if "n_jobs" in m.get_params():
        m.set_params(n_jobs=4)                  # thread count only; results unchanged
    return m


class Chain:
    """The design-time prediction sequence of one pool, fitted on one training part."""

    def __init__(self, recs, df, comp_cols, full_cols, tr):
        self.comp, self.full = comp_cols, full_cols
        t = df.iloc[tr]
        h = t[t["hardness_GPa"] > 0]
        self.V = fresh(recs["V"]).fit(t[comp_cols], t["vol_per_atom"].values)
        self.E = fresh(recs["E"]).fit(t[full_cols], np.log10(t["youngs_modulus_GPa"].values))
        self.H = fresh(recs["H"]).fit(h[full_cols], h["hardness_GPa"].values)

    def inputs(self, te, mode):
        """Model inputs for the rows te: database structure, or predicted as at design time."""
        X = te[self.full].copy()
        vpa = self.V.predict(te[self.comp])
        if mode == "design":
            X["vol_per_atom"] = vpa
            X["density"] = AMU_PER_A3_TO_GCC * te["mean_mass"].values / vpa
        return X, vpa

    def predict(self, te, mode):
        """log10 E, hardness and volume per atom for the rows te."""
        X, vpa = self.inputs(te, mode)
        return self.E.predict(X), self.H.predict(X), vpa


TOL = {"E": 1e-9, "V": 1e-9, "H": 0.25}


def check_released(df, comp_cols, full_cols, recs):
    """Refit every clone on the whole pool and compare with the released model.

    The gradient-boosting models and the random-forest volume model come back
    bit-identical. The hardness forests do not: in 45 of the 300 trees of the full-pool
    model two features give exactly the same impurity decrease at a node of a few
    samples, and which one is taken depends on floating-point summation order on the
    machine, so predictions differ by at most 0.08 GPa (full pool) and 0.17 GPa
    (dentin) against mean absolute errors of 2.6 to 3.9 GPa. TOL allows that and no more.
    """
    c = Chain(recs, df, comp_cols, full_cols, np.arange(len(df)))
    X = df[full_cols]
    worst = {}
    for k, fitted, cols in (("E", c.E, full_cols), ("H", c.H, full_cols), ("V", c.V, comp_cols)):
        assert type(fitted) is type(recs[k]["model"]), k
        assert list(recs[k]["features"]) == list(cols), k
        worst[k] = float(np.max(np.abs(fitted.predict(df[cols]) - recs[k]["model"].predict(df[cols]))))
    if any(worst[k] > TOL[k] for k in worst):
        raise SystemExit("refit does not reproduce the released models: %s" % worst)
    return worst
