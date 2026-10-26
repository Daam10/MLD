import argparse
import subprocess
import sys
from pathlib import Path

STEPS = [("media", "extract_media"), ("asr", "transcribe"), ("dialog", "build_dialog"), ("speech", "audio_features"),
         ("pose", "run_pose"), ("breathing", "run_breathing"), ("bio", "process_bio"), ("face", "run_face"),
         ("video", "process_video"), ("bio_final", "process_bio"), ("analysis", "analyze")]
names = [n for n, _ in STEPS]
ap = argparse.ArgumentParser()
ap.add_argument("--from", dest="start", choices=names)
ap.add_argument("--only", choices=names)
args = ap.parse_args()
run = [args.only] if args.only else names[names.index(args.start or names[0]):]
for name, script in STEPS:
    if name in run:
        print(f"== {name}", flush=True)
        subprocess.run([sys.executable, str(Path(__file__).parent / f"{script}.py")], check=True)
