import warnings

import numpy as np
import pandas as pd
from scipy import stats

TOPIC_CONDITIONS = ["honest", "prepared_lie", "unprepared_lie"]


def normalize_within_subject(df, cols, ref_mask):
    out = df.copy()
    out[cols] = out[cols].astype(float)
    for _, g in df.groupby("sid"):
        ref = g.loc[ref_mask.loc[g.index], cols]
        ref = ref if len(ref) >= 3 else g[cols]
        med = ref.median()
        mad = (ref - med).abs().median() * 1.4826
        out.loc[g.index, cols] = (g[cols] - med) / mad.where(mad > 1e-9, g[cols].std()).replace(0, np.nan)
    return out


def residualize_on_length(df, cols, length_col):
    out = df.copy()
    out[cols] = out[cols].astype(float)
    for _, g in df.groupby("sid"):
        x = np.log(g[length_col].clip(lower=0.3).to_numpy(float))
        for c in cols:
            y = g[c].to_numpy(float)
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 8 and np.std(x[ok]) > 0:
                b, a = np.polyfit(x[ok], y[ok], 1)
                out.loc[g.index, c] = y - (a + b * x)
    return out


def preprocess_features(df, cols):
    norm = normalize_within_subject(df, cols, df["condition"] == "baseline")
    norm["answer_dur_s"] = df["answer_dur"]
    return residualize_on_length(norm, cols, "answer_dur_s")


def fdr_bh(p):
    p = np.asarray(p, float)
    q, ok = np.full_like(p, np.nan), np.isfinite(p)
    order = np.argsort(p[ok])
    ranked = np.minimum.accumulate((p[ok][order] * ok.sum() / (np.arange(ok.sum()) + 1))[::-1])[::-1]
    res = np.empty(ok.sum())
    res[order] = np.minimum(ranked, 1)
    q[ok] = res
    return q


def rank_biserial(diff):
    diff = diff[np.isfinite(diff) & (diff != 0)]
    if len(diff) == 0:
        return np.nan
    r = stats.rankdata(np.abs(diff))
    return float((r[diff > 0].sum() - r[diff < 0].sum()) / r.sum())


def _lie_truth(df, label, c):
    lt = df.dropna(subset=[label]).groupby(["sid", label])[c].median().unstack()
    return (lt[1] - lt[0]).dropna().to_numpy() if {0, 1} <= set(lt.columns) else np.array([])


def subject_level_tests(df, cols, label):
    rows = []
    for c in cols:
        row = {"feature": c}
        full = df.groupby(["sid", "condition"])[c].median().unstack().reindex(columns=TOPIC_CONDITIONS).dropna()
        if len(full) >= 4:
            row["friedman_p"] = stats.friedmanchisquare(*[full[k] for k in TOPIC_CONDITIONS]).pvalue
        diff = _lie_truth(df, label, c)
        if len(diff) >= 4:
            row.update(wilcoxon_p=stats.wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0, effect_rb=rank_biserial(diff),
                       median_diff=float(np.median(diff)), n_subjects=len(diff))
        rows.append(row)
    res = pd.DataFrame(rows)
    for p in ("friedman_p", "wilcoxon_p"):
        if p in res:
            res[p.replace("_p", "_q")] = fdr_bh(res[p])
    return res.sort_values("wilcoxon_p")


def mixed_model(df, feature, moderator, label):
    import statsmodels.formula.api as smf
    d = df[["sid", label, feature, moderator]].dropna().rename(columns={label: "y"})
    if d["sid"].nunique() < 4 or d["y"].nunique() < 2:
        return {}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = smf.mixedlm(f"Q('{feature}') ~ y * Q('{moderator}')", d, groups=d["sid"]).fit(method="lbfgs")
    except Exception:
        return {}
    key = next(k for k in fit.params.index if k.startswith("y:"))
    return {"feature": feature, "moderator": moderator, "beta_lie": fit.params["y"], "p_lie": fit.pvalues["y"],
            "beta_interaction": fit.params[key], "p_interaction": fit.pvalues[key], "n": len(d)}


def temperament_markers(df, cols, label, top=8):
    rows = []
    for grp, g in df.dropna(subset=["temperament", label]).groupby("temperament"):
        for c in cols:
            diff = _lie_truth(g, label, c)
            if len(diff) >= 2:
                rows.append({"temperament": grp, "feature": c, "n_subjects": len(diff), "effect_rb": rank_biserial(diff),
                             "median_diff": float(np.median(diff))})
    res = pd.DataFrame(rows)
    if res.empty:
        return res
    res["abs_effect"] = res["effect_rb"].abs()
    return res.sort_values(["temperament", "abs_effect", "median_diff"], ascending=[True, False, False]).groupby("temperament").head(top)
