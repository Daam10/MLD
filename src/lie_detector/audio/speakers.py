import re
from dataclasses import asdict, dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression

SR = 16000
SENT_END = re.compile(r"[.?!…]$")
INTERVIEWER_STARTS = ("хорошо", "понял", "понятно", "спасибо", "отлично", "если мы говорим", "а если мы говорим",
                      "а вот", "но вот", "ну вот", "тогда", "давайте", "расскажите")
SHORT_ANSWERS = {"да", "нет", "ну да", "наверное", "не знаю", "да, конечно"}


@dataclass
class Unit:
    start: float
    end: float
    text: str
    n_words: int
    role: str = "?"
    p_interviewer: float = 0.5
    words: list = field(default_factory=list)

    @property
    def dur(self):
        return self.end - self.start


def _words(result):
    words = [dict(w) for seg in result["segments"] for w in seg.get("words", []) if w.get("word", "").strip()]
    for i, w in enumerate(words):
        if "start" not in w:
            prev = next((words[j]["end"] for j in range(i - 1, -1, -1) if "end" in words[j]), 0.0)
            nxt = next((words[j]["start"] for j in range(i + 1, len(words)) if "start" in words[j]), prev)
            w["start"], w["end"] = prev, max(prev, nxt)
    return words


def build_units(result, pause_s=0.6):
    units, buf, words = [], [], _words(result)
    for i, w in enumerate(words):
        buf.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        if nxt is None or SENT_END.search(w["word"].strip()) or nxt["start"] - w["end"] > pause_s:
            units.append(Unit(buf[0]["start"], buf[-1]["end"], " ".join(x["word"].strip() for x in buf), len(buf),
                              words=[(x["start"], x["end"], x["word"].strip()) for x in buf]))
            buf = []
    return units


def text_prior(u):
    t = u.text.lower().strip()
    return (1.5 if t.endswith("?") and u.n_words >= 4 else 0.0) + (1.0 if t.startswith(INTERVIEWER_STARTS) else 0.0) - \
        (1.0 if re.sub(r"[^\w\s,-]", "", t).strip() in SHORT_ANSWERS else 0.0)


class UnitEmbedder:
    def __init__(self, hf_token, device="cuda", model="pyannote/speaker-diarization-community-1"):
        import torch
        from pyannote.audio import Inference, Model
        self.inference = Inference(Model.from_pretrained(model, subfolder="embedding", token=hf_token), window="whole",
                                   device=torch.device(device))

    def __call__(self, audio, units, min_dur=0.4):
        import torch
        embs = np.full((len(units), 256), np.nan, dtype=np.float32)
        for i, u in enumerate(units):
            if u.dur >= min_dur:
                wav = torch.from_numpy(audio[int(max(0, u.start - 0.05) * SR):int((u.end + 0.05) * SR)][None].astype(np.float32))
                e = np.asarray(self.inference({"waveform": wav, "sample_rate": SR})).reshape(-1)
                embs[i, :len(e)] = e / (np.linalg.norm(e) + 1e-9)
        return embs


def seed_labels(units, embs):
    has, y = ~np.isnan(embs).any(1), np.full(len(units), -1)
    for i, u in enumerate(units):
        q = u.text.strip().endswith("?")
        if has[i] and q and u.n_words >= 4 and u.dur >= 1.0:
            y[i] = 1
        elif has[i] and not q and u.n_words >= 6 and u.dur >= 1.5 and text_prior(u) <= 0:
            y[i] = 0
    return y


def fit_role_model(embs, y):
    tr = y >= 0
    return LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced").fit(embs[tr], y[tr]) if len(set(y[tr])) == 2 else None


def _continues(prev, nxt):
    return not SENT_END.search(prev.text.strip()) and (nxt.text.strip()[:1].islower() or nxt.start - prev.end < 0.25)


def viterbi(units, emission):
    em = np.stack([-emission / 2, emission / 2], 1)
    score, back = em[0].copy(), np.zeros((len(units), 2), int)
    for i in range(1, len(units)):
        prev = units[i - 1]
        trans = np.zeros((2, 2))
        trans[0, 1] = trans[1, 0] = -(3.0 if _continues(prev, units[i]) else 0.3)
        if prev.text.strip().endswith("?") and prev.n_words >= 3:
            trans[1, 1] -= 1.0
        cand = score[:, None] + trans
        back[i], score = cand.argmax(0), cand.max(0) + em[i]
    path = np.zeros(len(units), int)
    path[-1] = score.argmax()
    for i in range(len(units) - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def attribute(units, embs, model):
    has = ~np.isnan(embs).any(1)
    ac = np.zeros(len(units))
    if model is not None:
        ac[has] = np.clip(model.decision_function(embs[has]), -6, 6)
    logit = np.array([text_prior(u) for u in units]) + np.clip(np.array([u.dur for u in units]) / 1.5, 0.2, 1.0) * ac
    for u, l, s in zip(units, logit, viterbi(units, logit)):
        u.p_interviewer, u.role = float(1 / (1 + np.exp(-l))), "I" if s == 1 else "R"
    return units


def units_to_records(units):
    return [asdict(u) for u in units]
