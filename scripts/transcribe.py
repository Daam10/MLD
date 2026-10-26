import json

from lie_detector.audio.transcribe import Transcriber, json_default, to_srt
from lie_detector.cli import session_args
from lie_detector.config import interim_dir, load_config, processed_dir

args, todo = session_args()
todo = [sid for sid in todo if args.force or not (processed_dir(sid) / "transcript.json").exists()]
tr = Transcriber(load_config()["asr"]) if todo else None
for sid in todo:
    res = tr(interim_dir(sid) / "audio_16k.wav")
    (processed_dir(sid) / "transcript.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=json_default), encoding="utf-8")
    (processed_dir(sid) / "transcript_raw.srt").write_text(to_srt(res["segments"]), encoding="utf-8")
    print(sid, flush=True)
