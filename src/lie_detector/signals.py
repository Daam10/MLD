import numpy as np
import pandas as pd
from scipy import signal


def bandpass(x, fs, lo, hi, order=2):
    return signal.sosfiltfilt(signal.butter(order, [lo, hi], btype="band", fs=fs, output="sos"), x, axis=0)


def lowpass(x, fs, hi, order=2):
    return signal.sosfiltfilt(signal.butter(order, hi, btype="low", fs=fs, output="sos"), x, axis=0)


def notch(x, fs, f0=50.0, harmonics=3, q=30):
    for k in range(1, harmonics + 1):
        if k * f0 < fs / 2:
            b, a = signal.iirnotch(k * f0, q, fs)
            x = signal.filtfilt(b, a, x)
    return x


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def interp_gaps(x, limit=None):
    return pd.DataFrame(x).interpolate(limit=limit, limit_direction="both").to_numpy().squeeze()


def spectral_rate(x, fs, lo, hi, window_s=30, step_s=5):
    n, hop = int(window_s * fs), int(step_s * fs)
    f = np.fft.rfftfreq(8 * n, 1 / fs)
    m = (f >= lo) & (f <= hi)
    ts, rates = [], []
    for a in range(0, len(x) - n + 1, hop):
        spec = np.abs(np.fft.rfft(x[a:a + n] * np.hanning(n), 8 * n)) ** 2
        ts.append((a + n / 2) / fs)
        rates.append(60 * f[m][spec[m].argmax()])
    return np.array(ts), np.array(rates)
