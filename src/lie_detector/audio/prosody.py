import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

VOWELS = re.compile(r"[аеёиоуыэюяaeiou]", re.I)


def syllables(text):
    return len(VOWELS.findall(text))


def dialogue_turns(dialog, max_gap=1.5):
    turns = []
    for u in dialog:
        t = turns[-1] if turns else None
        if t and t["role"] == u["role"] and u["start"] - t["end"] <= max_gap:
            t["end"], t["text"] = u["end"], t["text"] + " " + u["text"]
            t["words"] += u["words"]
            t["unit_ids"].append(u["id"])
        else:
            turns.append({**{k: u[k] for k in ("role", "start", "end", "text", "topic", "condition")},
                          "words": list(u["words"]), "unit_ids": [u["id"]]})
    return turns


class EnergyVad:
    def __init__(self, audio, sr, hop_s=0.01):
        hop = int(hop_s * sr)
        n = len(audio) // hop
        self.db = 10 * np.log10((audio[:n * hop].reshape(n, hop).astype(np.float64) ** 2).mean(1) + 1e-12)
        self.hop_s = hop_s
        lo, hi = np.percentile(self.db, [10, 90])
        self.thr = lo + 0.35 * (hi - lo)

    def longest_silence(self, t0, t1, min_s=0.05):
        a, b = int(t0 / self.hop_s), int(t1 / self.hop_s)
        best, start = (0, 0), None
        for i, s in enumerate(np.append(self.db[a:b] < self.thr, False) if b > a else []):
            if s and start is None:
                start = i
            elif not s and start is not None:
                best, start = max(best, (start, i), key=lambda r: r[1] - r[0]), None
        return ((a + best[0]) * self.hop_s, (a + best[1]) * self.hop_s) if (best[1] - best[0]) * self.hop_s >= min_s else None


def timing_features(words, question_end):
    starts, ends = np.array([w[0] for w in words]), np.array([w[1] for w in words])
    gaps = starts[1:] - ends[:-1]
    pauses = gaps[gaps > 0.25]
    dur = ends[-1] - starts[0]
    n_syl = syllables(" ".join(w[2] for w in words))
    per = lambda v: v / dur if dur > 0 else np.nan
    return {"latency": starts[0] - question_end if question_end is not None else np.nan, "answer_dur": dur,
            "n_words": len(words), "n_syllables": n_syl, "pause_total": pauses.sum(), "pause_n": len(pauses),
            "pause_ratio": per(pauses.sum()), "syll_rate": n_syl / max(dur - pauses.sum(), 1e-3),
            "speech_rate_sps": per(n_syl), "wpm": per(60 * len(words))}


def praat_features(snd, f0_min, f0_max):
    import parselmouth
    from parselmouth.praat import call
    out = {}
    f0 = snd.to_pitch_ac(time_step=0.01, pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array["frequency"]
    f0 = f0[f0 > 0]
    if len(f0) >= 5:
        st = 12 * np.log2(f0 / 100.0)
        out.update(f0_mean=f0.mean(), f0_median=np.median(f0), f0_sd_st=st.std(), f0_range_st=np.ptp(np.percentile(st, [5, 95])),
                   f0_slope_st_s=np.polyfit(np.arange(len(st)) * 0.01, st, 1)[0])
    pp = call(snd, "To PointProcess (periodic, cc)", f0_min, f0_max)
    try:
        out["jitter_local"] = call(pp, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
        out["shimmer_local"] = call([snd, pp], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    except parselmouth.PraatError:
        pass
    hv = snd.to_harmonicity_cc(time_step=0.01, minimum_pitch=f0_min).values
    if (hv > -200).any():
        out["hnr"] = float(hv[hv > -200].mean())
    iv = snd.to_intensity(minimum_pitch=f0_min).values.ravel()
    out.update(intensity_mean=iv.mean(), intensity_sd=iv.std(), intensity_cv_amp=np.std(10 ** (iv / 20)) / np.mean(10 ** (iv / 20)))
    fm = snd.to_formant_burg(time_step=0.01, max_number_of_formants=5, maximum_formant=5000 if f0_min < 90 else 5500)
    for k in (1, 2, 3):
        v = np.array([fm.get_value_at_time(k, t) for t in fm.ts()])
        if np.isfinite(v).any():
            out[f"F{k}_mean"] = v[np.isfinite(v)].mean()
    return out


def _make_smile():
    import opensmile
    root = Path(opensmile.__file__).parent / "core" / "config"
    if not str(root).isascii():
        dst = Path(os.environ.get("LOCALAPPDATA", Path.home())) / "lie_detector" / "opensmile_config"
        if not dst.exists():
            shutil.copytree(root, dst)
        root = dst

    class Smile(opensmile.Smile):
        default_config_root = property(lambda self: str(root))

    return Smile(feature_set=f"{root}/egemaps/v02/eGeMAPSv02.conf", feature_level=opensmile.FeatureLevel.Functionals)


class AcousticAnalyzer:
    def __init__(self, sex):
        self.smile = _make_smile()
        self.f0_range = (75, 300) if sex == "M" else (120, 500)

    def __call__(self, audio, sr, start, end):
        import parselmouth
        seg = audio[int(start * sr):int(end * sr)].astype(np.float64)
        if len(seg) < 0.3 * sr:
            return {}
        out = praat_features(parselmouth.Sound(seg, sampling_frequency=sr), *self.f0_range)
        return {**out, **{f"eg_{k}": float(v) for k, v in self.smile.process_signal(seg, sr).iloc[0].items()}}


def answer_table(dialog, audio, sr, sex):
    turns, an, vad, rows = dialogue_turns(dialog), AcousticAnalyzer(sex), EnergyVad(audio, sr), []
    for k, t in enumerate(turns):
        if t["role"] != "R" or not t["words"]:
            continue
        prev = turns[k - 1] if k and turns[k - 1]["role"] == "I" else None
        row = {"turn": k, "start": t["start"], "end": t["end"], "topic": t["topic"], "condition": t["condition"],
               "unit_ids": t["unit_ids"], "question": prev["text"] if prev else None, "text": t["text"],
               "is_question_answer": bool(prev and prev["text"].strip().endswith("?")),
               **timing_features(t["words"], prev["end"] if prev else None)}
        gap = vad.longest_silence(prev["words"][-1][0], t["words"][0][1]) if prev and prev["words"] else None
        row["latency"] = (gap[1] - gap[0] if gap else 0.0) if prev and prev["words"] else np.nan
        rows.append({**row, **an(audio, sr, t["start"], t["end"])})
    return pd.DataFrame(rows)


def anomaly_vector(df, baseline_mask, cols):
    base = df.loc[baseline_mask, cols]
    med = base.median()
    mad = ((base - med).abs().median() * 1.4826).pipe(lambda m: m.where(m > 1e-9, df[cols].std()))
    return ((df[cols] - med) / mad).add_prefix("z_")
