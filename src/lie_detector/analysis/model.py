import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier


class BalancedXGB(XGBClassifier):
    def fit(self, X, y, **kw):
        self.set_params(scale_pos_weight=(len(y) - np.sum(y)) / max(np.sum(y), 1.0))
        return super().fit(X, y, **kw)


def make_model(kind):
    if kind == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(C=0.1, class_weight="balanced", max_iter=5000))
    return BalancedXGB(n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5,
                       reg_lambda=1.0, tree_method="hist", n_jobs=1, random_state=0)


def add_temperament_interactions(X, temperament, cols):
    parts = [X] + [pd.DataFrame(X[cols].to_numpy() * (temperament == t).to_numpy(float)[:, None], index=X.index,
                                columns=[f"{c}__x_{t}" for c in cols]) for t in temperament.dropna().unique()]
    return pd.concat(parts, axis=1)


def usable_columns(X):
    return [c for c in X.columns if X[c].notna().sum() > 5 and X[c].nunique(dropna=True) > 1]


def _folds(X, y, groups):
    X = X.replace([np.inf, -np.inf], np.nan)
    for g in np.unique(groups):
        tr, te = groups != g, groups == g
        if len(np.unique(y[tr])) == 2:
            cols = usable_columns(X[tr])
            yield X.loc[tr, cols], y[tr], X.loc[te, cols], y[te], te


def loso_predict(X, y, groups, kind):
    p = np.full(len(y), np.nan)
    for Xtr, ytr, Xte, _, te in _folds(X, y, groups):
        p[te] = make_model(kind).fit(Xtr, ytr).predict_proba(Xte)[:, 1]
    return p


def scores(y, p, groups):
    ok = np.isfinite(p)
    y, p, groups = y[ok], p[ok], groups[ok]
    per = [roc_auc_score(y[groups == g], p[groups == g]) for g in np.unique(groups) if len(np.unique(y[groups == g])) == 2]
    return {"auc": roc_auc_score(y, p), "bal_acc": balanced_accuracy_score(y, p > 0.5),
            "auc_per_subject_mean": float(np.mean(per)) if per else np.nan, "n": int(ok.sum())}


def _null_auc(X, yp, groups, kind):
    with threadpool_limits(1):
        return scores(yp, loso_predict(X, yp, groups, kind), groups)["auc"]


def permutation_test(X, y, groups, kind, n_perm=200, seed=0):
    rng = np.random.default_rng(seed)
    observed = scores(y, loso_predict(X, y, groups, kind), groups)["auc"]
    perms = []
    for _ in range(n_perm):
        yp = y.copy()
        for g in np.unique(groups):
            yp[groups == g] = rng.permutation(yp[groups == g])
        perms.append(yp)
    null = np.array(Parallel(n_jobs=-1)(delayed(_null_auc)(X, yp, groups, kind) for yp in perms))
    return float((1 + np.sum(null >= observed)) / (1 + n_perm))


def loso_importance(X, y, groups, kind, seed=0):
    imps = [pd.Series(permutation_importance(make_model(kind).fit(Xtr, ytr), Xte, yte, scoring="roc_auc", n_repeats=10,
                                             random_state=seed, n_jobs=-1).importances_mean, index=Xtr.columns)
            for Xtr, ytr, Xte, yte, _ in _folds(X, y, groups) if len(np.unique(yte)) == 2]
    return pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)
