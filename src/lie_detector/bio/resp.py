import numpy as np
from scipy import interpolate, signal

from lie_detector.signals import bandpass, lowpass, notch, spectral_rate, zscore

RESP_BAND = (0.08, 0.7)


def clean_belt(raw, fs, out_fs=25.0):
    x = signal.resample_poly(lowpass(notch(raw.astype(np.float64), fs), fs, 2.0, order=4), int(out_fs), int(fs))
    return zscore(bandpass(x, out_fs, *RESP_BAND)), out_fs


def edr_from_rpeaks(ecg_clean, rpeaks, fs, out_fs=25.0):
    amp = ecg_clean[rpeaks]
    f = interpolate.interp1d(rpeaks / fs, amp, kind="cubic", bounds_error=False, fill_value=(amp[0], amp[-1]))
    return zscore(bandpass(f(np.arange(0, len(ecg_clean) / fs, 1 / out_fs)), out_fs, *RESP_BAND))


def breathing_rate(x, fs):
    return spectral_rate(x, fs, *RESP_BAND)
