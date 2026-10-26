import json

import numpy as np
import pandas as pd

from lie_detector.config import path, processed_dir, sessions
from lie_detector.labeling.sheets import read_sheet
from lie_detector.video.kinematics import segment_features

STYLE_TO_TEMPERAMENT = {"контролирующий": "choleric", "стимулирующий": "sanguine", "содействующий": "phlegmatic",
                        "координирующий": "melancholic"}
LENGTH_FAMILY = ("answer_dur", "n_words", "n_syllables", "pause_total", "pause_n")
META = {"turn", "start", "end", "y_self", "y_scen", "latency_aligned", "emo_balance", "style_strength",
        "is_question_answer", "in_study", *LENGTH_FAMILY}


def load_l34():
    df = pd.read_excel(path("l34_xlsx")).rename(columns={"Таб.": "tab_no", "Выраженность": "style_strength",
                                                         "Естественный": "style_natural", "Эмоц_Баланс": "emo_balance"})
    df = df.dropna(subset=["tab_no"]).astype({"tab_no": int})
    first = lambda s: s.split("/")[0].strip().lower().replace("содейстующий", "содействующий") if isinstance(s, str) else None
    df["temperament"] = df["style_natural"].map(lambda s: STYLE_TO_TEMPERAMENT.get(first(s)))
    return df.set_index("tab_no")[["style_natural", "style_strength", "emo_balance", "temperament"]]


def _window(t, q_end, a_end):
    return (t >= q_end - 12) & (t < q_end - 2), (t >= q_end - 2) & (t < a_end + 5)


def physio_features(hr_t, hr, belt_t, belt, q_end, a_end):
    out = {}
    pre, win = _window(hr_t, q_end, a_end)
    if pre.sum() >= 5 and win.sum() >= 5:
        base, w = hr[pre].mean(), hr[win]
        out.update(hr_mean=w.mean(), hr_delta=w.mean() - base, hr_peak_delta=w.max() - base, hr_min_delta=w.min() - base,
                   hr_sd=w.std(), rmssd_local=np.sqrt(np.mean(np.diff(60000 / w) ** 2)))
    bp, bw = _window(belt_t, q_end, a_end)
    if bp.sum() > 50 and bw.sum() > 50:
        out.update(resp_amp_ratio=np.std(belt[bw]) / (np.std(belt[bp]) + 1e-9), resp_baseline_shift=belt[bw].mean() - belt[bp].mean())
        fs, seg = 1 / np.median(np.diff(belt_t)), belt[bw] - belt[bw].mean()
        if len(seg) > 4 * fs:
            spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), 8 * len(seg))) ** 2
            f = np.fft.rfftfreq(8 * len(seg), 1 / fs)
            m = (f >= 0.08) & (f <= 0.7)
            out["resp_rate"] = 60 * f[m][spec[m].argmax()]
    return out


def pat_features(pat, q_end, a_end):
    t, v = pat["t_video"].to_numpy(), pat["pat"].to_numpy()
    pre, win = _window(t, q_end, a_end)
    if pre.sum() < 5 or win.sum() < 5:
        return {}
    return {"pat_delta_ms": 1000 * (np.median(v[win]) - np.median(v[pre])), "pat_min_delta_ms": 1000 * (np.min(v[win]) - np.median(v[pre]))}


def load_self_reports(sid, dialog):
    files = sorted(path("labels_dir").glob(f"{sid}_*.xlsx"))
    if not files:
        return {}
    df = read_sheet(files[0], assume_done=True)
    df = df[df["is_respondent"]]
    if np.mean([str(dialog.get(int(i), {}).get("text", "")).strip() == str(t).strip() for i, t in zip(df["id"], df["text"])]) < 0.8:
        by_text = {str(u["text"]).strip(): uid for uid, u in dialog.items()}
        df = df.assign(id=[by_text.get(str(t).strip(), -1) for t in df["text"]]).query("id >= 0")
    return {int(r.id): r.self_report for r in df.dropna(subset=["self_report"]).itertuples()}


def build_dataset():
    l34, rows = load_l34(), []
    for sid, s in sessions().items():
        d = processed_dir(sid)
        ans, sync, bio = pd.read_csv(d / "answers.csv"), json.loads((d / "sync.json").read_text()), np.load(d / "bio.npz")
        rp = bio["rpeaks_s"]
        rr = np.diff(rp)
        ok = (rr > 0.33) & (rr < 1.5)
        hr_t, hr = rp[1:][ok] - sync["offset"], 60 / rr[ok]
        belt_t = np.arange(len(bio["belt"])) / float(bio["resp_fs"]) - sync["offset"]
        kin = pd.read_parquet(d / "kinematics.parquet")
        pat = pd.read_csv(d / "pat.csv") if (d / "pat.csv").exists() else None
        dialog = {u["id"]: u for u in json.loads((d / "dialog.json").read_text(encoding="utf-8"))}
        reports = load_self_reports(sid, dialog)
        info = l34.loc[s.tab_no].to_dict() if s.tab_no in l34.index else {}
        for a in ans.to_dict("records"):
            q_end = a["start"] - (a["latency"] if np.isfinite(a["latency"]) else 0)
            marks = {reports.get(i) for i in json.loads(a["unit_ids"])} - {None}
            rows.append({"sid": sid, "sex": s.sex, "in_study": s.in_study, **a,
                         **physio_features(hr_t, hr, belt_t, bio["belt"], q_end, a["end"]),
                         **(pat_features(pat, q_end, a["end"]) if pat is not None else {}),
                         **{f"kin_{k}": v for k, v in segment_features(kin, a["start"], a["end"], 25.0).items()},
                         "self_report": "Ложь" if "Ложь" in marks else "Частично" if "Частично" in marks else "Правда" if marks else None,
                         **{k: info.get(k) for k in ("temperament", "style_natural", "style_strength", "emo_balance")}})
    df = pd.DataFrame(rows)
    df["y_self"] = df["self_report"].map({"Ложь": 1.0, "Частично": 1.0, "Правда": 0.0})
    df["y_scen"] = df["condition"].map({"prepared_lie": 1.0, "unprepared_lie": 1.0, "honest": 0.0, "baseline": 0.0})
    return df


def feature_columns(df):
    num = df.select_dtypes(include=[np.number]).columns
    return [c for c in num if c not in META and not c.startswith("z_") and df[c].notna().mean() > 0.6]
