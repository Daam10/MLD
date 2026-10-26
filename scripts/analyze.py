import argparse
import json

import numpy as np
import pandas as pd

from lie_detector.analysis import model as M
from lie_detector.analysis.dataset import build_dataset, feature_columns
from lie_detector.analysis.stats import (TOPIC_CONDITIONS, fdr_bh, mixed_model, preprocess_features, subject_level_tests,
                                          temperament_markers)
from lie_detector.config import REPO_ROOT, processed_dir

HYPOTHESES = [("latency", +1), ("pause_ratio", +1), ("syll_rate", -1), ("f0_mean", +1), ("anomaly_norm", +1),
              ("hr_delta", +1), ("rmssd_local", -1), ("resp_amp_ratio", -1), ("kin_head_nod_power", -1),
              ("kin_rhand_speed_mean", -1)]
MODALITIES = {"speech": lambda c: not c.startswith(("kin_", "hr_", "resp_", "rmssd", "pat_")),
              "physiology": lambda c: c.startswith(("hr_", "resp_", "rmssd", "pat_")),
              "kinematics": lambda c: c.startswith("kin_")}


def evaluate(name, frame, label, cols, n_perm):
    X, y, g = frame[cols], frame[label].to_numpy().astype(int), frame["sid"].to_numpy()
    base = {"labels": name, "n_answers": len(frame), "n_lies": int(y.sum()), "n_subjects": frame["sid"].nunique()}
    rows, preds = [], None
    for kind in ("logreg", "gboost"):
        p = M.loso_predict(X, y, g, kind)
        rows.append({**base, "model": kind, "features": "multimodal", **M.scores(y, p, g),
                     "perm_p": M.permutation_test(X, y, g, kind, n_perm)})
        preds = p
    Xt = M.add_temperament_interactions(X, frame["temperament"], cols)
    rows.append({**base, "model": "gboost", "features": "multimodal+temperament", **M.scores(y, M.loso_predict(Xt, y, g, "gboost"), g)})
    for mod, pick in MODALITIES.items():
        rows.append({**base, "model": "gboost", "features": mod, **M.scores(y, M.loso_predict(frame[[c for c in cols if pick(c)]], y, g, "gboost"), g)})
    return rows, preds


def confirmatory(frame, label):
    st = subject_level_tests(frame, [f for f, _ in HYPOTHESES], label).set_index("feature")
    res = pd.DataFrame([{"feature": f, "expected": "+" if s > 0 else "-", **st.loc[f, ["median_diff", "effect_rb", "wilcoxon_p"]],
                         "as_expected": bool(np.sign(st.loc[f, "median_diff"]) == s)} for f, s in HYPOTHESES])
    res["q"] = fdr_bh(res["wilcoxon_p"])
    return res


def write_predictions(norm, cols, frame, label, p):
    for sid, g in frame.assign(p_lie=p).groupby("sid"):
        g.assign(t=(g["start"] + g["end"]) / 2)[["start", "end", "t", "p_lie"]].to_csv(processed_dir(sid) / "predictions.csv", index=False)
    use = M.usable_columns(frame[cols].replace([np.inf, -np.inf], np.nan))
    final = M.make_model("gboost").fit(frame[use], frame[label].astype(int))
    for sid, g in norm[~norm["sid"].isin(frame["sid"].unique())].groupby("sid"):
        g.assign(t=(g["start"] + g["end"]) / 2, p_lie=final.predict_proba(g[use].replace([np.inf, -np.inf], np.nan))[:, 1])[
            ["start", "end", "t", "p_lie"]].to_csv(processed_dir(sid) / "predictions.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--stats-only", action="store_true")
    args = ap.parse_args()
    out = REPO_ROOT / "outputs"
    out.mkdir(exist_ok=True)
    df = build_dataset()
    df.to_csv(out / "dataset.csv", index=False)
    cols = feature_columns(df[df["in_study"]])
    norm = preprocess_features(df, cols)
    study = norm[norm["in_study"]]
    sets = {"self": (study[study["y_self"].notna()], "y_self"),
            "scenario": (study[study["condition"].isin(TOPIC_CONDITIONS)], "y_scen")}
    d = df[df["y_self"].notna()]
    d.groupby(["sid", "condition"])["y_self"].agg(share_self_lie="mean", n_answers="count").reset_index().to_csv(
        out / "label_agreement.csv", index=False)
    for name, (frame, label) in sets.items():
        subject_level_tests(frame, cols, label).to_csv(out / f"stats_{name}.csv", index=False)
    frame, label = sets["self"]
    conf = confirmatory(frame, label)
    conf.to_csv(out / "confirmatory.csv", index=False)
    pd.DataFrame([r for c in cols for m in ("emo_balance", "style_strength") if (r := mixed_model(frame, c, m, label))]).to_csv(
        out / "mixed_models.csv", index=False)
    temperament_markers(frame, cols, label).to_csv(out / "temperament_markers.csv", index=False)
    hits = conf[conf["wilcoxon_p"] < 0.05]
    print(f"{len(frame)} answers, {int(frame[label].sum())} lies, {frame['sid'].nunique()} subjects, {len(cols)} features; "
          f"a-priori markers p<0.05: {', '.join(hits['feature'])}")
    if args.stats_only:
        return
    rows, preds = [], {}
    for name, (frame, label) in sets.items():
        r, preds[name] = evaluate(name, frame, label, cols, args.n_perm)
        rows += r
    models = pd.DataFrame(rows)
    models.to_csv(out / "models.csv", index=False)
    frame, label = sets["self"]
    M.loso_importance(frame[cols], frame[label].to_numpy().astype(int), frame["sid"].to_numpy(), "gboost").to_csv(
        out / "importance.csv", header=["auc_drop"])
    write_predictions(norm, cols, frame, label, preds["self"])
    (out / "summary.json").write_text(json.dumps({n: {"answers": len(f), "lies": int(f[l].sum()), "subjects": int(f["sid"].nunique())}
                                                 for n, (f, l) in sets.items()}, indent=2), encoding="utf-8")
    best = models[(models["model"] == "gboost") & (models["features"] == "multimodal")]
    print("\n".join(f"{r.labels}: AUC {r.auc:.2f}, permutation p={r.perm_p:.3f}" for r in best.itertuples()))


if __name__ == "__main__":
    main()
