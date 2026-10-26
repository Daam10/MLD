import numpy as np
import pandas as pd
from scipy import interpolate, signal

BANDS = {"ULF": (0.0, 0.003), "VLF": (0.003, 0.04), "LF": (0.04, 0.15), "HF": (0.15, 0.4)}


def clean_rr(beat_t):
    rr, rr_t = np.diff(beat_t), beat_t[1:]
    med = pd.Series(rr).rolling(9, center=True, min_periods=1).median().to_numpy()
    ok = (rr > 0.33) & (rr < 1.5) & (np.abs(rr - med) < 0.2 * med)
    return rr_t[ok], rr[ok]


def time_domain(rr):
    nn = rr * 1000.0
    d, hr = np.diff(nn), 60000.0 / nn
    sdnn = np.std(nn, ddof=1)
    sd1 = np.sqrt(0.5) * np.std(d, ddof=1) if len(d) > 1 else np.nan
    sd2 = np.sqrt(max(2 * sdnn ** 2 - sd1 ** 2, 0)) if np.isfinite(sd1) else np.nan
    lo = nn.min() // 50 * 50
    hist, edges = np.histogram(nn, bins=np.arange(lo, max(nn.max(), lo + 50) + 50, 50))
    mo, amo, mxdmn = (edges[hist.argmax()] + 25) / 1000.0, 100.0 * hist.max() / len(nn), (nn.max() - nn.min()) / 1000.0
    return {"HR_mean": hr.mean(), "HR_std": hr.std(ddof=1), "MeanNN": nn.mean(), "SDNN": sdnn,
            "RMSSD": np.sqrt(np.mean(d ** 2)), "SDSD": np.std(d, ddof=1), "pNN50": 100 * np.mean(np.abs(d) > 50),
            "CVNN": sdnn / nn.mean(), "SD1": sd1, "SD2": sd2, "SD1_SD2": sd1 / sd2 if sd2 else np.nan,
            "Mo": mo, "AMo": amo, "MxDMn": mxdmn, "SI_Baevsky": amo / (2 * mo * mxdmn) if mxdmn > 0 else np.nan}


def spectrum(rr_t, rr, t0, t1, fs=4.0):
    f = interpolate.interp1d(rr_t, rr * 1000.0, kind="cubic", bounds_error=False, fill_value=(rr[0] * 1000.0, rr[-1] * 1000.0))
    x = signal.detrend(f(np.arange(t0, t1, 1 / fs)))
    return signal.welch(x, fs=fs, window="hann", nperseg=len(x), detrend=False)


def band_powers(freqs, psd, resolvable_from):
    out = {}
    for name, (lo, hi) in BANDS.items():
        m = (freqs >= max(lo, resolvable_from)) & (freqs < hi)
        out[name] = np.trapezoid(psd[m], freqs[m]) if hi > resolvable_from and m.sum() > 1 else np.nan
    m = (freqs >= resolvable_from) & (freqs < 0.4)
    lf, hf, hfm = out["LF"], out["HF"], (freqs >= 0.15) & (freqs < 0.4)
    return {**out, "TP": np.trapezoid(psd[m], freqs[m]), "LF_HF": lf / hf if hf else np.nan,
            "LFnu": 100 * lf / (lf + hf) if lf + hf else np.nan, "HFnu": 100 * hf / (lf + hf) if lf + hf else np.nan,
            "HF_peak_Hz": freqs[hfm][psd[hfm].argmax()]}


def sliding_hrv(beat_t, window_s=120, step_s=5, t_start=None, min_coverage=0.8):
    rr_t, rr = clean_rr(np.asarray(beat_t, float))
    rows = []
    for a in np.arange(rr_t[0] if t_start is None else t_start, rr_t[-1] - window_s + 1e-9, step_s):
        m = (rr_t >= a) & (rr_t < a + window_s)
        row = {"t": a + window_s / 2, "t0": a, "t1": a + window_s, "n_beats": int(m.sum()), "coverage": rr[m].sum() / window_s}
        if row["coverage"] >= min_coverage and m.sum() > 20:
            row.update(time_domain(rr[m]), **band_powers(*spectrum(rr_t[m], rr[m], a, a + window_s), 1.0 / window_s))
        rows.append(row)
    return pd.DataFrame(rows)


def session_hrv(beat_t, t_start, t_end):
    rr_t, rr = clean_rr(np.asarray(beat_t, float))
    m = (rr_t >= t_start) & (rr_t < t_end)
    return {**time_domain(rr[m]), **band_powers(*spectrum(rr_t[m], rr[m], t_start, t_end), 0.0), "duration_s": t_end - t_start}
