import pandas as pd
import soundfile as sf

from lie_detector.audio.prosody import anomaly_vector, answer_table
from lie_detector.cli import session_args
from lie_detector.config import interim_dir, processed_dir
from lie_detector.labeling.dialog import load_dialog

ANOMALY_COLS = ["latency", "pause_ratio", "syll_rate", "wpm", "f0_mean", "f0_sd_st", "f0_range_st", "jitter_local",
                "shimmer_local", "hnr", "intensity_mean", "intensity_sd", "F1_mean", "F2_mean", "F3_mean",
                "eg_slopeV0-500_sone_mean", "eg_alphaRatioV_sma3nz_amean", "eg_hammarbergIndexV_sma3nz_amean",
                "eg_spectralFlux_sma3_amean"]

args, todo = session_args("--anomaly-only")
for sid, s in todo.items():
    out = processed_dir(sid) / "answers.csv"
    if args.anomaly_only:
        df = pd.read_csv(out).filter(regex=r"^(?!z_|anomaly_norm$)")
    else:
        audio, sr = sf.read(interim_dir(sid) / "audio_16k.wav", dtype="float32")
        df = answer_table(load_dialog(sid), audio, sr, s.sex)
    z = anomaly_vector(df, df["condition"] == "baseline", [c for c in ANOMALY_COLS if c in df])
    df.join(z).assign(anomaly_norm=(z.clip(-5, 5) ** 2).mean(axis=1) ** 0.5).to_csv(out, index=False)
    print(sid, flush=True)
