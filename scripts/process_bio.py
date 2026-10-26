import json

import numpy as np
import pandas as pd
import soundfile as sf
from scipy import signal

from lie_detector.audio.transcribe import json_default
from lie_detector.bio.ecg import process_ecg, rr_quality
from lie_detector.bio.hrv import session_hrv, sliding_hrv
from lie_detector.bio.ppg import hrv_agreement, process_ppg, pulse_arrival_time, riiv
from lie_detector.bio.resp import RESP_BAND, breathing_rate, clean_belt, edr_from_rpeaks
from lie_detector.cli import session_args
from lie_detector.config import interim_dir, load_config, processed_dir
from lie_detector.io.bcrx import load_session_bio
from lie_detector.signals import bandpass, zscore
from lie_detector.sync.align import event_bursts, keep_pulse_correction, load_sync, offset_by_breathing, save_sync

HRV_INDICES = ["HR_mean", "MeanNN", "SDNN", "RMSSD", "pNN50", "SD1", "SD2", "SI_Baevsky", "VLF", "LF", "HF", "TP", "LF_HF", "LFnu", "HFnu"]
cfg = load_config()["bio"]


def align_clocks(sid, rec, belt, rfs, video_dur, bio_dur):
    bursts = event_bursts(rec.events_s)
    off, r = offset_by_breathing(belt, np.load(processed_dir(sid) / "breath_video.npz")["breath"], rfs, bio_dur - video_dur)
    if r < 0.2 and bursts:
        off = bursts[0]
    return {"offset": off, "method": "video_breathing_xcorr" if r >= 0.2 else "events",
            "coarse_offset": min(bursts, key=lambda b: abs(b - off)) if bursts else off, "score": r, "video_duration": video_dur}


def with_time(df, t, offset, video_dur):
    tv = np.asarray(t) - offset
    return df.assign(t_video=tv, phase=np.where(tv < 0, "pre", np.where(tv > video_dur, "post", "interview")))


def rate_mae(a, b, fs):
    k = min(len(a), len(b))
    return float(np.mean(np.abs(breathing_rate(a[:k], fs)[1] - breathing_rate(b[:k], fs)[1])))


def contact_ppg(sid, rec, rpeaks_s, hrv_ecg, belt, rfs, offset, video_dur):
    pfs, d = rec.fs_of["ppg"], processed_dir(sid)
    ppg = process_ppg(rec.signals["ppg"], pfs)
    hrv = sliding_hrv(ppg.peaks_s, cfg["hrv_window_s"], cfg["hrv_step_s"], t_start=0.0)
    hrv = with_time(hrv, hrv["t"], offset, video_dur)
    hrv.to_csv(d / "hrv_ppg.csv", index=False)
    inter = lambda h: h[h["phase"] == "interview"]
    hrv_agreement(inter(hrv_ecg), inter(hrv), HRV_INDICES).assign(sid=sid).to_csv(d / "ppg_vs_ecg.csv", index=False)
    pat_t, pat = pulse_arrival_time(rpeaks_s, ppg.feet_s)
    pd.DataFrame({"t": pat_t, "t_video": pat_t - offset, "pat": pat}).to_csv(d / "pat.csv", index=False)
    rv = riiv(rec.signals["ppg"], ppg.valid, pfs, rfs)
    af = zscore(bandpass(signal.resample_poly(rec.signals["airflow"].astype(np.float64), int(rfs), int(rec.fs_of["airflow"])), rfs, *RESP_BAND))
    t1 = np.arange(len(rec.signals["hr_oximeter"])) / rec.fs_of["hr_oximeter"]
    sec = np.arange(10, t1[-1] - 10, 1.0)
    ho = np.interp(sec, t1, rec.signals["hr_oximeter"].astype(float))
    he = pd.Series(np.interp(sec, rpeaks_s[1:], 60 / np.diff(rpeaks_s))).rolling(5, center=True, min_periods=1).mean().to_numpy()
    ok, sp = (ho > 30) & (ho < 220), rec.signals["spo2"][(rec.signals["spo2"] > 70) & (rec.signals["spo2"] <= 100)]
    summary = {"ppg_valid_frac": float(ppg.valid.mean()), "n_pulses": len(ppg.peaks), "pat_median_ms": float(1000 * np.median(pat)),
               "riiv_vs_belt_rate_mae": rate_mae(belt, rv, rfs), "airflow_vs_belt_rate_mae": rate_mae(belt, af, rfs),
               "oximeter_hr_vs_ecg_mae": float(np.mean(np.abs(ho[ok] - he[ok]))), "spo2_median": float(np.median(sp))}
    return {"ppg": signal.resample_poly(ppg.clean, 1, 2).astype(np.float32), "ppg_fs": pfs / 2, "ppg_peaks_s": ppg.peaks_s,
            "ppg_feet_s": ppg.feet_s, "riiv": rv.astype(np.float32), "airflow": af.astype(np.float32)}, summary


args, todo = session_args()
for sid, s in todo.items():
    rec, d = load_session_bio(s), processed_dir(sid)
    fs, n = rec.fs, min(len(rec.signals[k]) for k in ("ecg", "resp", "gsr"))
    ecg = process_ecg(rec.signals["ecg"][:n], fs)
    belt, rfs = clean_belt(rec.signals["resp"][:n], fs)
    edr = edr_from_rpeaks(ecg.clean, ecg.rpeaks, fs, rfs)
    video_dur = sf.info(str(interim_dir(sid) / "audio_16k.wav")).duration
    sync = keep_pulse_correction(align_clocks(sid, rec, belt, rfs, video_dur, n / fs), load_sync(sid))
    save_sync(sid, sync)
    off, rpeaks_s = sync["offset"], ecg.rpeaks / fs
    hrv = sliding_hrv(rpeaks_s, cfg["hrv_window_s"], cfg["hrv_step_s"], t_start=0.0)
    hrv = with_time(hrv, hrv["t"], off, video_dur)
    hrv.to_csv(d / "hrv_ecg.csv", index=False)
    tb, rb = breathing_rate(belt, rfs)
    _, re_ = breathing_rate(edr, rfs)
    with_time(pd.DataFrame({"t": tb, "rate_belt": rb, "rate_edr": re_[:len(tb)]}), tb, off, video_dur).to_csv(d / "resp_rate.csv", index=False)
    extra, ppg_summary = {}, None
    gsr = rec.signals["gsr"][:n].astype(np.float64)
    if (gsr < 99999).mean() >= 0.9:
        extra.update(gsr=signal.resample_poly(np.where(gsr < 99999, gsr, np.median(gsr[gsr < 99999])), 1, int(fs // 10)).astype(np.float32), gsr_fs=10.0)
    if "ppg" in rec.signals:
        ppg_extra, ppg_summary = contact_ppg(sid, rec, rpeaks_s, hrv, belt, rfs, off, video_dur)
        extra.update(ppg_extra)
    np.savez_compressed(d / "bio.npz", ecg=signal.resample_poly(ecg.clean, 1, 4).astype(np.float32), ecg_fs=250.0, rpeaks_s=rpeaks_s,
                        belt=belt.astype(np.float32), edr=edr.astype(np.float32), resp_fs=rfs, **extra)
    summary = {"ecg_quality": rr_quality(ecg.rr_s), "ecg_inverted": ecg.inverted, "pre_rest_s": off,
               "hrv_pre": session_hrv(rpeaks_s, 5.0, off - 2) if off > 60 else None,
               "hrv_interview": session_hrv(rpeaks_s, off, off + video_dur),
               "belt_vs_edr_rate_mae": float(np.mean(np.abs(rb - re_[:len(rb)]))), "ppg": ppg_summary}
    (d / "bio_summary.json").write_text(json.dumps(summary, indent=1, default=json_default), encoding="utf-8")
    print(sid, flush=True)
