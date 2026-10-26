import numpy as np
import pandas as pd

from lie_detector.cli import session_args
from lie_detector.config import processed_dir
from lie_detector.sync.align import apply_pulse_correction, load_sync, pulse_lag_vs_ecg, save_sync
from lie_detector.video import rppg
from lie_detector.video.kinematics import frame_features

WIN, STEP, MAX_JUMP, OUTPUT_METHOD = 20.0, 1.0, 6.0, "POS"


def hr_on_grid(beats, t_centres, win=WIN):
    out = np.full(len(t_centres), np.nan)
    for i, c in enumerate(t_centres):
        b = beats[(beats >= c - win / 2) & (beats < c + win / 2)]
        rr = np.diff(b)[(np.diff(b) > 0.33) & (np.diff(b) < 1.5)]
        if len(b) > 3 and len(rr):
            out[i] = 60 / rr.mean()
    return out


def agreement(est, ref):
    m = np.isfinite(est) & np.isfinite(ref)
    e, r = est[m], ref[m]
    return {"MAE": float(np.mean(np.abs(e - r))), "RMSE": float(np.sqrt(np.mean((e - r) ** 2))), "r": float(np.corrcoef(e, r)[0, 1]),
            "within5": float(np.mean(np.abs(e - r) <= 5)), "bias": float(np.mean(e - r)), "n": int(m.sum())}


def process_rppg(sid):
    d = processed_dir(sid)
    f, pose, bio, sync = np.load(d / "face_rgb.npz"), np.load(d / "pose.npz"), np.load(d / "bio.npz"), load_sync(sid)
    fs = float(f["fps"])
    rgb = rppg.fill_gaps(rppg.combine_rois(f["rgb"], f["counts"]))
    ok = np.isfinite(rgb).all(1)
    rgb[~ok] = np.nanmean(rgb, axis=0)
    rgb = rppg.normalize_illumination(rgb, f["bg"], fs)
    motion = np.gradient(pd.DataFrame(np.nanmean(pose["kpts"][:, 23:91], axis=1)).interpolate(limit_direction="both").to_numpy(), axis=0)[:len(rgb)]
    refs = {"ECG": bio["rpeaks_s"] - sync["offset"]}
    if "ppg_peaks_s" in bio.files:
        refs["PPG"] = bio["ppg_peaks_s"] - sync["offset"]
    rows, out = [], None
    for name, fn in rppg.METHODS.items():
        pulse = rppg.suppress_motion(rppg.bandpass(fn(rgb, fs), fs, *rppg.HR_BAND), motion, fs)
        pulse[~ok] = 0
        track = rppg.spectrogram_hr(pulse, fs, WIN, STEP, MAX_JUMP)
        grid = {k: hr_on_grid(v, track["t"].to_numpy()) for k, v in refs.items()}
        rows += [{"reference": k, "method": name, **agreement(track["hr"].to_numpy(), g)} for k, g in grid.items()]
        if name == OUTPUT_METHOD:
            out = (pulse, track, grid["ECG"])
    pulse, track, ref = out
    if "PPG" in refs:
        rows.append({"reference": "ECG", "method": "contact_PPG", **agreement(hr_on_grid(refs["PPG"], track["t"].to_numpy()), ref)})
    pd.DataFrame(rows).to_csv(d / "rppg_eval.csv", index=False)
    np.savez_compressed(d / "rppg.npz", pulse=pulse.astype(np.float32), fs=fs, method=OUTPUT_METHOD, hr_t=track["t"].to_numpy(),
                        hr=track["hr"].to_numpy(), hr_ecg=ref, peaks_s=rppg.pulse_peaks(pulse, fs, track),
                        resp=rppg.respiration_from_pulse(rppg.green(rgb, fs), fs).astype(np.float32), valid=ok)
    save_sync(sid, apply_pulse_correction(sync, *pulse_lag_vs_ecg(rppg.bandpass(pulse, fs, *rppg.HR_BAND), fs, bio["rpeaks_s"], sync["offset"])))


args, todo = session_args()
for sid in todo:
    p = np.load(processed_dir(sid) / "pose.npz")
    frame_features(p["kpts"], p["scores"], float(p["fps"])).to_parquet(processed_dir(sid) / "kinematics.parquet")
    process_rppg(sid)
    print(sid, flush=True)
