import numpy as np
import pandas as pd
from scipy import signal

from lie_detector.signals import lowpass
from lie_detector.video.pose import BODY, FACE, FEET, L_HAND, R_HAND

CONF = 0.4


def _pt(kp, sc, idx):
    k, s = kp[:, idx], sc[:, idx]
    if k.ndim == 3:
        return np.nanmean(np.where((s > CONF)[..., None], k, np.nan), axis=1)
    return np.where((s > CONF)[:, None], k, np.nan)


def _speed(p, fs):
    filled = pd.DataFrame(p).interpolate(limit=12, limit_direction="both").to_numpy()
    return np.linalg.norm(np.gradient(filled, axis=0) * fs, axis=1)


def _smooth(x, fs):
    ok = np.isfinite(x)
    if ok.sum() <= 30:
        return x
    return np.where(ok, lowpass(pd.Series(x).interpolate(limit_direction="both").to_numpy(), fs, 6.0), np.nan)


def frame_features(kpts, scores, fs):
    P = lambda name: _pt(kpts, scores, BODY[name])
    ls, rs = P("l_shoulder"), P("r_shoulder")
    sh, hip = np.nanmean(np.stack([ls, rs]), 0), np.nanmean(np.stack([P("l_hip"), P("r_hip")]), 0)
    trunk = sh - hip
    L = np.nanmedian(np.linalg.norm(trunk, axis=1))
    ears = np.nanmean(np.stack([P("l_ear"), P("r_ear")]), 0)
    face = _pt(kpts, scores, FACE)
    face_c = np.where(np.isnan(face), P("nose"), face)
    lw, rw = P("l_wrist"), P("r_wrist")
    lhand, rhand = _pt(kpts, scores, L_HAND), _pt(kpts, scores, R_HAND)
    lhand, rhand = np.where(np.isnan(lhand), lw, lhand), np.where(np.isnan(rhand), rw, rhand)
    la, ra, lk, rk = P("l_ankle"), P("r_ankle"), P("l_knee"), P("r_knee")
    com = np.nanmean(np.where((scores[:, :17] > CONF)[..., None], kpts[:, :17], np.nan), axis=1)
    face_size = np.nanmedian(np.linalg.norm(np.nanmax(kpts[:, FACE], 1) - np.nanmin(kpts[:, FACE], 1), axis=1))
    near_face = lambda h: np.linalg.norm(h - face_c, axis=1) < 1.2 * face_size
    df = pd.DataFrame({
        "t": np.arange(len(kpts)) / fs,
        "spine_angle_deg": _smooth(np.degrees(np.arctan2(trunk[:, 0], -trunk[:, 1])), fs),
        "trunk_len_norm": _smooth(np.linalg.norm(trunk, axis=1) / L, fs),
        "head_forward_norm": _smooth((ears[:, 0] - sh[:, 0]) / L, fs),
        "com_shift_x_norm": (com[:, 0] - hip[:, 0]) / L,
        "shoulder_tilt_deg": np.degrees(np.arctan2(ls[:, 1] - rs[:, 1], ls[:, 0] - rs[:, 0] + 1e-9)),
        "head_speed": _speed(face_c - sh, fs) / L,
        "trunk_speed": _speed(sh, fs) / L,
        "lhand_speed": _speed(lhand, fs) / L,
        "rhand_speed": _speed(rhand, fs) / L,
        "lfoot_speed": _speed(np.where(np.isnan(la), lk, la), fs) / L,
        "rfoot_speed": _speed(np.where(np.isnan(ra), rk, ra), fs) / L,
        "self_touch": (near_face(lhand) | near_face(rhand)).astype(float),
        "hands_together": (np.linalg.norm(lhand - rhand, axis=1) / L < 0.25).astype(float),
        "arms_crossed": ((np.linalg.norm(lw - P("r_elbow"), axis=1) / L < 0.35) &
                         (np.linalg.norm(rw - P("l_elbow"), axis=1) / L < 0.35)).astype(float),
        "head_x": _smooth((face_c[:, 0] - sh[:, 0]) / L, fs),
        "head_y": _smooth((face_c[:, 1] - sh[:, 1]) / L, fs),
        "shoulder_y": _smooth((hip[:, 1] - sh[:, 1]) / L, fs),
        "feet_y": _pt(kpts, scores, FEET)[:, 1] / L,
    })
    return df


def _welch(x, fs):
    x = pd.Series(x).interpolate(limit_direction="both").to_numpy()
    if np.isnan(x).any() or len(x) < fs * 2:
        return None, None
    return signal.welch(signal.detrend(x), fs=fs, nperseg=min(len(x), int(4 * fs)))


def band_power(x, fs, lo, hi):
    f, p = _welch(x, fs)
    return np.nan if f is None else float(np.trapezoid(p[(f >= lo) & (f < hi)], f[(f >= lo) & (f < hi)]))


def dominant_freq(x, fs, lo, hi):
    f, p = _welch(x, fs)
    return np.nan if f is None else float(f[(f >= lo) & (f < hi)][p[(f >= lo) & (f < hi)].argmax()])


def segment_features(ff, t0, t1, fs):
    s = ff[(ff["t"] >= t0) & (ff["t"] < t1)]
    if len(s) < 5:
        return {}
    out = {}
    for c in ("spine_angle_deg", "trunk_len_norm", "head_forward_norm", "com_shift_x_norm", "shoulder_tilt_deg"):
        out[f"{c}_mean"], out[f"{c}_delta"] = s[c].mean(), s[c].iloc[-5:].mean() - s[c].iloc[:5].mean()
    for c in ("head_speed", "trunk_speed", "lhand_speed", "rhand_speed", "lfoot_speed", "rfoot_speed"):
        out[f"{c}_mean"], out[f"{c}_p90"] = s[c].mean(), s[c].quantile(0.9)
    for c in ("self_touch", "hands_together", "arms_crossed"):
        out[f"{c}_frac"] = s[c].mean()
    col = lambda c: s[c].to_numpy()
    out.update(head_nod_power=band_power(col("head_y"), fs, 0.5, 3.0), head_turn_power=band_power(col("head_x"), fs, 0.5, 3.0),
               head_nod_freq=dominant_freq(col("head_y"), fs, 0.5, 3.0), foot_tap_power=band_power(col("feet_y"), fs, 1.0, 5.0),
               shrug_power=band_power(col("shoulder_y"), fs, 0.6, 2.0))
    return out
