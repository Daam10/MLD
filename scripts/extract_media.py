from lie_detector.cli import session_args
from lie_detector.config import interim_dir, load_config
from lie_detector.io.raw import open_video, pipe_to_ffmpeg

args, todo = session_args()
m = load_config()["media"]
for sid, s in todo.items():
    out = interim_dir(sid)
    if (out / "proxy.mp4").exists() and not args.force:
        continue
    pcm = lambda sr, name: ["-map", "0:a:0", "-ac", "1", "-ar", str(sr), "-c:a", "pcm_s16le", str(out / name)]
    ff = pcm(m["audio_sr_asr"], "audio_16k.wav") + pcm(m["audio_sr_prosody"], "audio_48k.wav") + [
        "-map", "0:v:0", "-vf", f"yadif=0,scale=-2:{m['proxy_height']}", "-r", str(m["proxy_fps"]), "-c:v", "h264_nvenc",
        "-preset", "p5", "-cq", "23", "-g", "25", "-an", str(out / "proxy.mp4")]
    if pipe_to_ffmpeg(open_video(s.video, s.video_zip), ff, log=out / "ffmpeg.log"):
        raise SystemExit(f"{sid}: ffmpeg failed, see {out / 'ffmpeg.log'}")
    print(sid, flush=True)
