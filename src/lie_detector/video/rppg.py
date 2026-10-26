import numpy as np
import pandas as pd
from scipy import signal

from lie_detector import signals as S

HR_BAND = (0.7, 3.0)


def bandpass(x, fs, lo, hi, order=3):
    return S.bandpass(x, fs, lo, hi, order)


def fill_gaps(x, max_gap=50):
    return pd.DataFrame(x).interpolate(limit=max_gap, limit_direction="both").to_numpy()


def combine_rois(rgb, counts):
    w = counts.astype(float)[..., None]
    den = w.sum(1)
    return np.nansum(np.nan_to_num(rgb) * w, axis=1) / np.where(den > 0, den, np.nan)


def normalize_illumination(rgb, bg, fs):
    if bg is None or np.isnan(bg).all():
        return rgb
    b = S.lowpass(fill_gaps(bg, 10 ** 6), fs, 0.5)
    return rgb / (b / np.nanmean(b, axis=0))


def _windows(n, l, step):
    return ((a, a + l) for a in range(0, max(n - l, 0) + 1, step))


def pos(rgb, fs, win_s=1.6):
    h, proj = np.zeros(len(rgb)), np.array([[0, 1, -1], [-2, 1, 1]])
    for a, b in _windows(len(rgb), int(win_s * fs), 1):
        s = (rgb[a:b] / (rgb[a:b].mean(0) + 1e-9)) @ proj.T
        p = s[:, 0] + (s[:, 0].std() / (s[:, 1].std() + 1e-9)) * s[:, 1]
        h[a:b] += p - p.mean()
    return h


def chrom(rgb, fs, win_s=1.6):
    l = int(win_s * fs)
    l += l % 2
    h, hann = np.zeros(len(rgb)), np.hanning(l)
    for a, b in _windows(len(rgb), l, l // 2):
        c = rgb[a:b] / (rgb[a:b].mean(0) + 1e-9)
        xf = bandpass(3 * c[:, 0] - 2 * c[:, 1], fs, *HR_BAND, order=2)
        yf = bandpass(1.5 * c[:, 0] + c[:, 1] - 1.5 * c[:, 2], fs, *HR_BAND, order=2)
        p = xf - (xf.std() / (yf.std() + 1e-9)) * yf
        h[a:b] += hann * (p - p.mean())
    return h


def omit(rgb, fs, win_s=1.6):
    h = np.zeros(len(rgb))
    for a, b in _windows(len(rgb), int(win_s * fs), 1):
        c = (rgb[a:b] / (rgb[a:b].mean(0) + 1e-9)).T
        s = np.linalg.qr(c)[0][:, 0:1]
        y = ((np.eye(3) - s @ s.T) @ c)[1]
        h[a:b] += y - y.mean()
    return h


def green(rgb, fs):
    g = rgb[:, 1]
    return -(g - pd.Series(g).rolling(int(fs), center=True, min_periods=1).mean().to_numpy())


METHODS = {"GREEN": green, "CHROM": chrom, "POS": pos, "OMIT": omit}


def suppress_motion(pulse, motion, fs):
    m = np.column_stack([bandpass(np.nan_to_num(motion), fs, *HR_BAND), np.ones(len(motion))])
    return pulse - m @ np.linalg.lstsq(m, pulse, rcond=None)[0]


def spectrogram_hr(pulse, fs, win_s=10.0, step_s=1.0, jump_bpm=12.0):
    nper = int(win_s * fs)
    f, t, Sx = signal.spectrogram(pulse, fs=fs, window="hann", nperseg=nper, noverlap=nper - int(step_s * fs),
                                  nfft=8 * nper, detrend="constant", mode="psd")
    m = (f >= HR_BAND[0]) & (f <= HR_BAND[1])
    bpm, Sx = 60 * f[m], Sx[m]
    logp = np.log(Sx / (Sx.sum(0, keepdims=True) + 1e-12) + 1e-9)
    trans = -((bpm[:, None] - bpm[None, :]) / jump_bpm) ** 2
    score, back = logp[:, 0].copy(), np.zeros(Sx.shape, int)
    for k in range(1, Sx.shape[1]):
        cand = score[:, None] + trans
        back[:, k], score = cand.argmax(0), cand.max(0) + logp[:, k]
    path = np.zeros(Sx.shape[1], int)
    path[-1] = score.argmax()
    for k in range(Sx.shape[1] - 1, 0, -1):
        path[k - 1] = back[path[k], k]
    near = [(np.abs(bpm - bpm[i]) <= 6) | (np.abs(bpm - 2 * bpm[i]) <= 6) for i in path]
    snr = [10 * np.log10(Sx[n, k].sum() / (Sx[~n, k].sum() + 1e-12)) for k, n in enumerate(near)]
    return pd.DataFrame({"t": t, "hr": bpm[path], "hr_argmax": bpm[Sx.argmax(0)], "snr_db": snr})


def pulse_peaks(pulse, fs, hr_track):
    x = bandpass(pulse, fs, *HR_BAND)
    hr = np.interp(np.arange(len(x)) / fs, hr_track["t"], hr_track["hr"])
    return signal.find_peaks(x, distance=max(int(0.6 * fs * 60 / np.percentile(hr, 90)), 1), prominence=0.2 * np.std(x))[0] / fs


def respiration_from_pulse(pulse_raw, fs):
    return bandpass(pulse_raw, fs, 0.1, 0.6, order=2)
