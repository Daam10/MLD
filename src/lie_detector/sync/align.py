import json

import numpy as np
from scipy.signal import find_peaks

from lie_detector.config import processed_dir

PTT_FACE_S = 0.2
PULSE_KEYS = ("pulse_correction_s", "offset_before_pulse", "rppg_pulse_lag_s", "rppg_pulse_lag_r")


def load_sync(sid: str) -> dict:
    p = processed_dir(sid) / "sync.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def save_sync(sid: str, sync: dict) -> None:
    (processed_dir(sid) / "sync.json").write_text(json.dumps(sync, indent=2), encoding="utf-8")


def keep_pulse_correction(sync: dict, old: dict) -> dict:
    if "pulse_correction_s" in old and abs(old.get("offset_before_pulse", -1e9) - sync["offset"]) < 0.05:
        sync.update({k: old[k] for k in PULSE_KEYS if k in old})
        sync["offset"] += old["pulse_correction_s"]
        sync["method"] += "+rppg_pulse"
    return sync


def apply_pulse_correction(sync: dict, lag: float, r: float, min_r: float = 0.25) -> dict:
    sync.update(rppg_pulse_lag_s=lag, rppg_pulse_lag_r=r)
    if r >= min_r and "pulse_correction_s" not in sync:
        sync.update(pulse_correction_s=pulse_offset_correction(lag), offset_before_pulse=sync["offset"])
        sync["offset"] += sync["pulse_correction_s"]
        sync["method"] += "+rppg_pulse"
    return sync


def _z(x):
    return (x - np.nanmean(x)) / np.nanstd(x)


def refine_by_xcorr(bio_feat, video_feat, fs, coarse, max_lag_s=5.0, step_s=None):
    bf, vf, best = _z(bio_feat), _z(video_feat), (coarse, -np.inf)
    for off in np.arange(coarse - max_lag_s, coarse + max_lag_s + 1e-9, step_s or 1 / fs):
        i0 = int(round(off * fs))
        seg = bf[max(i0, 0):i0 + len(vf)]
        n = min(len(seg), len(vf))
        m = np.isfinite(seg[:n]) & np.isfinite(vf[:n])
        if i0 >= 0 and m.sum() >= 10 * fs:
            r = np.corrcoef(seg[:n][m], vf[:n][m])[0, 1]
            if r > best[1]:
                best = (float(off), float(r))
    return best


def event_bursts(events_s, gap_s=2.0):
    ev = np.sort(np.asarray(events_s, float))
    return [float(ev[0])] + [float(b) for a, b in zip(ev[:-1], ev[1:]) if b - a > gap_s] if len(ev) else []


def _robust(x):
    med = np.median(x)
    return np.clip((x - med) / (np.median(np.abs(x - med)) * 1.4826 + 1e-12), -3, 3)


def chunk_consensus(belt, video, fs, offsets, chunk_s=60.0):
    b, v, n = _robust(belt), _robust(video), int(chunk_s * fs)
    chunks = [(i, _z(v[i:i + n])) for i in range(0, len(v) - n + 1, n)]
    out = np.full(len(offsets), np.nan)
    for k, off in enumerate(offsets):
        rs = []
        for i, c in chunks:
            j = int(round(off * fs)) + i
            s = b[j:j + n]
            if j >= 0 and len(s) == n:
                rs.append(float(np.dot((s - s.mean()) / (s.std() + 1e-12), c)) / n)
        if len(rs) >= max(1, len(chunks) // 2):
            out[k] = np.mean(rs)
    return out


def offset_by_breathing(belt, video_breath, fs, max_offset, fine_step=0.04):
    grid = np.arange(0, max(max_offset, 0) + 1e-9, 0.2)
    coarse = grid[np.nanargmax(chunk_consensus(belt, video_breath, fs, grid))]
    fine = np.arange(coarse - 1, coarse + 1 + 1e-9, fine_step)
    sf = chunk_consensus(belt, video_breath, fs, fine)
    return float(fine[np.nanargmax(sf)]), float(np.nanmax(sf))


def pulse_lag_vs_ecg(pulse, fs, rpeaks_s, offset, max_lag=2.5):
    train = np.zeros(len(pulse))
    idx = np.round((rpeaks_s - offset) * fs).astype(int)
    train[idx[(idx >= 0) & (idx < len(train))]] = 1
    train = np.convolve(train, np.exp(-0.5 * (np.arange(-6, 7) / 2.0) ** 2), "same")
    x = np.nan_to_num(pulse)
    lags = np.arange(-max_lag - 0.5, max_lag + 0.5 + 1e-9, 1 / fs)
    cc = np.array([np.corrcoef(*((x[k:], train[:len(train) - k]) if k >= 0 else (x[:k], train[-k:])))[0, 1]
                   for k in np.round(lags * fs).astype(int)])
    pk = find_peaks(cc)[0]
    pk = pk[np.abs(lags[pk]) <= max_lag]
    k = pk[np.argmax(cc[pk])] if len(pk) else int(np.argmax(cc))
    return float(lags[k]), float(cc[k])


def pulse_offset_correction(lag, ptt=PTT_FACE_S):
    return ptt - lag
