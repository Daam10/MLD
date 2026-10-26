from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from lie_detector.config import interim_dir, processed_dir, sessions

TOPIC_COLORS = {"pre": "#9e9e9e", "intro": "#90caf9", "alcohol": "#ffb74d", "conflict": "#e57373", "deception": "#ba68c8", "outro": "#a5d6a7"}
CONDITION_LABELS = {"baseline": "база", "honest": "правда", "prepared_lie": "подгот. ложь", "unprepared_lie": "неподгот. ложь", None: ""}


@dataclass
class Series:
    t: np.ndarray
    y: np.ndarray
    label: str
    color: str = "#1976d2"


@dataclass
class SessionData:
    sid: str
    duration: float
    video_path: str
    audio_path: str
    dialog: list[dict]
    answers: pd.DataFrame | None
    signals: dict[str, Series] = field(default_factory=dict)
    hr: dict[str, Series] = field(default_factory=dict)
    hrv: pd.DataFrame | None = None
    pose: dict | None = None
    stress: Series | None = None
    meta: dict = field(default_factory=dict)

    def utterance_at(self, t):
        return next((i for i, u in enumerate(self.dialog) if u["start"] <= t <= u["end"] + 0.3), None)

    def answer_at(self, t):
        if self.answers is None or self.answers.empty:
            return None
        a = self.answers
        hit = a[(a["start"] <= t) & (a["end"] + 1.0 >= t)]
        before = a[a["start"] <= t]
        return hit.iloc[0] if len(hit) else before.iloc[-1] if len(before) else None

    def hrv_at(self, t):
        return None if self.hrv is None else self.hrv.loc[(self.hrv["t_video"] - t).abs().idxmin()]

    def topic_spans(self):
        spans = []
        for u in self.dialog:
            if spans and spans[-1][2] == u["topic"]:
                spans[-1] = (spans[-1][0], u["end"], u["topic"])
            else:
                spans.append((u["start"], u["end"], u["topic"]))
        return spans

    def question_starts(self):
        return [u["start"] for u in self.dialog if u["role"] == "I" and u["text"].strip().endswith("?")]


def _z(x, ref):
    ref = ref[np.isfinite(ref)]
    if len(ref) < 3:
        return np.full_like(x, np.nan, dtype=float)
    return (x - np.median(ref)) / (np.median(np.abs(ref - np.median(ref))) * 1.4826 + 1e-9)


def load_session(sid):
    pdir, idir = processed_dir(sid), interim_dir(sid)
    dialog = json.loads((pdir / "dialog.json").read_text(encoding="utf-8"))
    answers = pd.read_csv(pdir / "answers.csv") if (pdir / "answers.csv").exists() else None
    sync = json.loads((pdir / "sync.json").read_text(encoding="utf-8")) if (pdir / "sync.json").exists() else None
    d = SessionData(sid, sync["video_duration"] if sync else dialog[-1]["end"], str(idir / "proxy.mp4"), str(idir / "audio_16k.wav"),
                    dialog, answers, meta={"scenario": sessions()[sid].scenario, "sync": sync})
    if sync and (pdir / "bio.npz").exists():
        off, b = sync["offset"], np.load(pdir / "bio.npz")
        t_of = lambda k, fs: np.arange(len(b[k])) / float(b[fs]) - off
        tr = t_of("belt", "resp_fs")
        d.signals["ecg"] = Series(t_of("ecg", "ecg_fs"), b["ecg"], "ЭКГ", "#d32f2f")
        d.signals["belt"] = Series(tr, b["belt"], "Дыхание (пояс)", "#388e3c")
        rp = b["rpeaks_s"]
        d.hr["ecg"] = Series(rp[1:] - off, 60 / np.diff(rp), "ЧСС ЭКГ", "#d32f2f")
        if "ppg" in b.files:
            d.signals["ppg"] = Series(t_of("ppg", "ppg_fs"), b["ppg"], "ФПГ (палец)", "#1565c0")
            pk = b["ppg_peaks_s"]
            ok = (np.diff(pk) > 0.33) & (np.diff(pk) < 1.5)
            d.hr["ppg"] = Series(pk[1:][ok] - off, 60 / np.diff(pk)[ok], "ЧСС ФПГ", "#1565c0")
            d.signals["riiv"] = Series(tr[:len(b["riiv"])], b["riiv"], "Дыхание (ФПГ)", "#5e35b1")
        if "gsr" in b.files:
            d.signals["gsr"] = Series(t_of("gsr", "gsr_fs"), b["gsr"], "КГР (сырой сигнал)", "#00838f")
        if "airflow" in b.files:
            d.signals["airflow"] = Series(tr[:len(b["airflow"])], b["airflow"], "Поток воздуха", "#6d4c41")
        if (pdir / "hrv_ecg.csv").exists():
            d.hrv = pd.read_csv(pdir / "hrv_ecg.csv")
    if (pdir / "rppg.npz").exists():
        r = np.load(pdir / "rppg.npz")
        tv = np.arange(len(r["pulse"])) / float(r["fs"])
        d.signals["rppg"] = Series(tv, r["pulse"], "rPPG (лицо)", "#7b1fa2")
        d.signals["rppg_resp"] = Series(tv, r["resp"], "Дыхание (лицо)", "#00897b")
        d.hr["rppg"] = Series(r["hr_t"], r["hr"], "ЧСС rPPG", "#7b1fa2")
    if (pdir / "breath_video.npz").exists():
        v = np.load(pdir / "breath_video.npz")
        d.signals["video_breath"] = Series(np.arange(len(v["breath"])) / float(v["fs"]), v["breath"], "Дыхание (грудь, видео)", "#f57c00")
    if (pdir / "pose.npz").exists():
        p = np.load(pdir / "pose.npz")
        d.pose = {"kpts": p["kpts"], "scores": p["scores"], "fps": float(p["fps"])}
    d.stress = stress_index(d)
    return d


def stress_index(d):
    pred = processed_dir(d.sid) / "predictions.csv"
    if pred.exists():
        p = pd.read_csv(pred)
        return Series(p["t"].to_numpy(), 100 * p["p_lie"].to_numpy(), "P(ложь), %", "#c62828")
    if d.hrv is None:
        return None
    h = d.hrv
    base = h["t_video"] < next((u["start"] for u in d.dialog if u["topic"] != "intro"), d.duration)
    comps = [_z(h["HR_mean"].to_numpy(), h.loc[base, "HR_mean"].to_numpy()),
             _z(np.log(h["SI_Baevsky"].to_numpy()), np.log(h.loc[base, "SI_Baevsky"].to_numpy()))]
    t = h["t_video"].to_numpy()
    if d.answers is not None and "anomaly_norm" in d.answers:
        a = d.answers
        comps.append(np.interp(t, (a["start"] + a["end"]).to_numpy() / 2, a["anomaly_norm"].to_numpy()) - 1)
    return Series(t, 100 / (1 + np.exp(-np.nanmean(np.vstack(comps), axis=0))), "Индекс стресса", "#c62828")
