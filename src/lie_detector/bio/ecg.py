from dataclasses import dataclass

import neurokit2 as nk
import numpy as np

from lie_detector.signals import notch

SATURATION_V = 0.1874


@dataclass
class EcgResult:
    fs: float
    clean: np.ndarray
    rpeaks: np.ndarray
    valid: np.ndarray
    inverted: bool
    n_corrected: int

    @property
    def rr_s(self):
        return np.diff(self.rpeaks) / self.fs


def saturation_mask(x, fs, pad_s=1.0):
    bad = np.abs(x) >= SATURATION_V
    if bad.any():
        bad = np.convolve(bad.astype(float), np.ones(int(pad_s * fs) * 2 + 1), "same") > 0
    return ~bad


def detect_inversion(clean, valid, fs, seg_s=120.0):
    n = int(seg_s * fs)
    ok = np.convolve(valid.astype(float), np.ones(n) / n, "valid") if len(valid) > n else np.array([1.0])
    start = int(np.argmax(ok)) if ok.max() > 0.99 else 0
    return bool(nk.ecg_invert(clean[start:start + n], sampling_rate=fs, force=False)[1])


def process_ecg(raw, fs) -> EcgResult:
    valid = saturation_mask(raw, fs)
    x = raw.astype(np.float64).copy()
    x[~valid] = np.median(x[valid])
    clean = nk.ecg_clean(notch(x, fs), sampling_rate=fs, method="neurokit")
    inverted = detect_inversion(clean, valid, fs)
    clean = -clean if inverted else clean
    peaks = np.asarray(nk.ecg_peaks(clean, sampling_rate=fs, method="neurokit")[1]["ECG_R_Peaks"])
    peaks = peaks[valid[peaks]]
    fixed = np.unique(np.asarray(nk.signal_fixpeaks(peaks, sampling_rate=fs, iterative=True, method="Kubios")[1], int))
    fixed = fixed[(fixed >= 0) & (fixed < len(clean))]
    fixed = fixed[valid[fixed]]
    return EcgResult(fs, clean, fixed, valid, inverted, int(len(np.setxor1d(peaks, fixed))))


def rr_quality(rr):
    jumps = np.abs(np.diff(rr)) / rr[1:]
    return {"n_beats": int(len(rr) + 1), "frac_physiological": float(((rr > 0.33) & (rr < 1.5)).mean()),
            "frac_jumps_gt20pct": float((jumps > 0.2).mean()), "hr_median_bpm": float(60 / np.median(rr))}
