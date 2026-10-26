import json
import re

import requests

from lie_detector.audio.speakers import attribute, build_units, units_to_records
from lie_detector.config import processed_dir

TOPICS = ("intro", "alcohol", "conflict", "deception", "outro")
SYSTEM = """Ты размечаешь расшифровку интервью психолога (интервьюер, I) с респондентом (R).
Структура интервью: вступление и знакомство (intro), затем три темы в произвольном порядке —
алкоголь (alcohol), конфликты (conflict), обман/ложь (deception) — и завершение беседы (outro).
Интервьюер задаёт вопросы, объявляет темы, говорит «Хорошо», «Понял», «Спасибо». Респондент отвечает,
иногда переспрашивает. Распознавание речи могло ошибиться в словах — ориентируйся на смысл.
Для каждой реплики дана подсказка акустической модели p_I — вероятность, что говорит интервьюер
(она ненадёжна на коротких репликах). Реплики иногда разрезаны посреди фразы — части одной фразы
принадлежат одному говорящему.
Ответь строго построчно в формате: <номер> <I|R> <тема>. Без пояснений, по одной строке на каждую реплику."""


def structure_dialog(units, endpoint, model, timeout=600):
    text = "\n".join(f"{i} [{u['start']:.1f}s p_I={u['p_interviewer']:.2f}] {u['text']}" for i, u in enumerate(units))
    resp = requests.post(f"{endpoint}/chat/completions", timeout=timeout, json={
        "model": model, "temperature": 0, "max_tokens": 12 * len(units) + 256,
        "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}]})
    resp.raise_for_status()
    return parse_answer(resp.json()["choices"][0]["message"]["content"], len(units))


def parse_answer(text, n):
    out = [None] * n
    for m in re.finditer(r"^\s*(\d+)\s*[:.)]?\s*([IR])\b\s*(\w+)?", text, re.M):
        if 0 <= int(m.group(1)) < n:
            topic = (m.group(3) or "").lower()
            out[int(m.group(1))] = {"role": m.group(2), "topic": topic if topic in TOPICS else None}
    for i in range(n):
        if out[i] is None:
            prev = next((out[j] for j in range(i - 1, -1, -1) if out[j]), None)
            out[i] = {"role": None, "topic": prev["topic"] if prev else None}
    return out


def smooth_topics(topics, min_run=3):
    t = list(topics)
    for i in range(len(t)):
        t[i] = t[i] if t[i] is not None else (t[i - 1] if i else "intro")
    runs, start = [], 0
    for i in range(1, len(t) + 1):
        if i == len(t) or t[i] != t[start]:
            runs.append((start, i))
            start = i
    for k, (a, b) in enumerate(runs):
        if b - a < min_run and 0 < k < len(runs) - 1:
            t[a:b] = [t[runs[k - 1][0]]] * (b - a)
    return t


def build_dialog(session, transcript, embs, model, llm_cfg, cache=None):
    recs = units_to_records(attribute(build_units(transcript), embs, model))
    llm = json.loads(cache.read_text(encoding="utf-8")) if cache and cache.exists() else None
    if llm is None or len(llm) != len(recs):
        llm = structure_dialog(recs, llm_cfg["endpoint"], llm_cfg["model"]) if llm_cfg else None
        if llm and cache:
            cache.write_text(json.dumps(llm, ensure_ascii=False), encoding="utf-8")
    topics = smooth_topics([a["topic"] for a in llm] if llm else [None] * len(recs))
    cond = session.condition_of_topic()
    for i, r in enumerate(recs):
        r["role_acoustic"] = r["role"]
        r["role"] = (llm[i]["role"] if llm else None) or r["role"]
        r["topic"], r["id"] = topics[i], i
        r["condition"] = None if r["role"] != "R" else "baseline" if r["topic"] == "intro" else cond.get(r["topic"])
    return recs


def load_dialog(sid):
    return json.loads((processed_dir(sid) / "dialog.json").read_text(encoding="utf-8"))
