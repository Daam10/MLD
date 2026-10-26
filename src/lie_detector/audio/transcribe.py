import numpy as np

SR = 16000


class Transcriber:
    def __init__(self, cfg, device="cuda"):
        import whisperx
        self.cfg, self.device = cfg, device
        self.asr = whisperx.load_model(cfg["model"], device, compute_type=cfg["compute_type"], language=cfg["language"],
                                       asr_options={"beam_size": 5})
        self.align_model, self.align_meta = whisperx.load_align_model(cfg["language"], device)

    def __call__(self, wav):
        import whisperx
        audio = whisperx.load_audio(str(wav), SR)
        res = self.asr.transcribe(audio, batch_size=self.cfg["batch_size"], language=self.cfg["language"])
        res = whisperx.align(res["segments"], self.align_model, self.align_meta, audio, self.device, return_char_alignments=False)
        return {**res, "duration": len(audio) / SR}


def ts(t):
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def to_srt(segments, names=None):
    out = []
    for i, s in enumerate(segments, 1):
        spk = s.get("role") or s.get("speaker")
        out += [str(i), f"{ts(s['start'])} --> {ts(s['end'])}", (f"[{(names or {}).get(spk, spk)}] " if spk else "") + s["text"].strip(), ""]
    return "\n".join(out)


def json_default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(type(o))
