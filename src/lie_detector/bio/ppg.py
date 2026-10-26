from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import signal

from lie_detector.signals import bandpass, zscore


@dataclass
class PpgResult:
    fs: float
    clean: np.ndarray
    peaks: np.ndarray
    feet: np.ndarray
    valid: np.ndarray

    @property
    def peaks_s(self):
        return self.peaks / self.fs

    @property
    def feet_s(self):
        return self.feet / self.fs


def valid_mask(x, fs, win_s=2.0):
    n = int(win_s * fs)
    sd = pd.Series(x).rolling(n, center=True, min_periods=n // 2).std().to_numpy()
    med = np.nanmedian(sd)
    lo, hi = np.nanpercentile(x, [0.5, 99.5])
    clipped = pd.Series(((x <= lo) | (x >= hi)).astype(float)).rolling(n, center=True, min_periods=1).mean().to_numpy() > 0.2
    return (sd > 0.1 * med) & (sd < 10 * med) & ~clipped


def process_ppg(raw, fs) -> PpgResult:
    x = raw.astype(np.float64)
    valid = valid_mask(x, fs)
    clean = bandpass(np.where(valid, x, np.nanmedian(x[valid]) if valid.any() else 0.0), fs, 0.5, 8.0, order=3)
    amp = pd.Series(np.abs(clean)).rolling(int(5 * fs), center=True, min_periods=1).quantile(0.9).to_numpy()
    peaks = signal.find_peaks(clean, distance=int(0.3 * fs), prominence=0.3 * np.median(amp))[0]
    peaks = peaks[valid[peaks]]
    d1 = np.gradient(clean)
    feet = np.array([max(p - int(0.35 * fs), 0) + int(np.argmax(d1[max(p - int(0.35 * fs), 0):p + 1])) for p in peaks], int)
    return PpgResult(fs, clean, peaks, feet, valid)


def pulse_arrival_time(rpeaks_s, feet_s, lo=0.08, hi=0.6):
    idx = np.searchsorted(feet_s, rpeaks_s + lo)
    ok = idx < len(feet_s)
    t, pat = rpeaks_s[ok], feet_s[idx[ok]] - rpeaks_s[ok]
    return t[pat <= hi], pat[pat <= hi]


def riiv(raw, valid, fs, out_fs=25.0):
    x = np.where(valid, raw, np.nanmedian(raw[valid]) if valid.any() else 0.0).astype(np.float64)
    return zscore(bandpass(signal.resample_poly(x, int(out_fs), int(fs)), out_fs, 0.08, 0.7))


def icc_a1(a, b):
    x = np.column_stack([a, b])
    n, k = x.shape
    gm = x.mean()
    msr = k * ((x.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((x.mean(0) - gm) ** 2).sum() / (k - 1)
    mse = ((x - x.mean(1, keepdims=True) - x.mean(0, keepdims=True) + gm) ** 2).sum() / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def hrv_agreement(ecg, ppg, indices):
    m = ecg.merge(ppg, on="t", suffixes=("_ecg", "_ppg"))
    rows = []
    for c in indices:
        a, b = m[f"{c}_ecg"].to_numpy(float), m[f"{c}_ppg"].to_numpy(float)
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() < 10:
            continue
        a, b = a[ok], b[ok]
        d = b - a
        rows.append({"index": c, "n_windows": int(ok.sum()), "mean_ecg": a.mean(), "mean_ppg": b.mean(), "bias": d.mean(),
                     "loa_low": d.mean() - 1.96 * d.std(ddof=1), "loa_high": d.mean() + 1.96 * d.std(ddof=1),
                     "rel_bias_pct": 100 * d.mean() / abs(a.mean()), "pearson_r": float(np.corrcoef(a, b)[0, 1]),
                     "icc": icc_a1(a, b)})
    return pd.DataFrame(rows)
