import json
import os

import numpy as np
import soundfile as sf

from lie_detector.audio.speakers import UnitEmbedder, build_units, fit_role_model, seed_labels
from lie_detector.audio.transcribe import json_default, to_srt
from lie_detector.cli import session_args
from lie_detector.config import hf_token, interim_dir, load_config, processed_dir, sessions
from lie_detector.labeling.dialog import build_dialog
from lie_detector.labeling.sheets import ROLE_LABELS, read_sheet, write_sheet

args, todo = session_args("--no-llm", "--rewrite-sheets")
llm = None if args.no_llm else {**load_config()["llm"], **({"endpoint": os.environ["LLM_ENDPOINT"]} if "LLM_ENDPOINT" in os.environ else {})}
embedder, data = None, {}
for sid in sessions():
    tr = json.loads((processed_dir(sid) / "transcript.json").read_text(encoding="utf-8"))
    units, p = build_units(tr), processed_dir(sid) / "unit_emb.npy"
    if not p.exists() or len(np.load(p)) != len(units):
        embedder = embedder or UnitEmbedder(hf_token())
        np.save(p, embedder(sf.read(interim_dir(sid) / "audio_16k.wav", dtype="float32")[0], units))
    data[sid] = (tr, units, np.load(p))
model = fit_role_model(np.concatenate([e for *_, e in data.values()]), np.concatenate([seed_labels(u, e) for _, u, e in data.values()]))
lab = processed_dir("labeling")
for sid, s in todo.items():
    d = processed_dir(sid)
    dialog = build_dialog(s, data[sid][0], data[sid][2], model, llm, cache=d / "llm_dialog.json")
    (d / "dialog.json").write_text(json.dumps(dialog, ensure_ascii=False, indent=1, default=json_default), encoding="utf-8")
    (d / "transcript.srt").write_text(to_srt(dialog, {"I": "Психолог", "R": "Респондент"}), encoding="utf-8")
    stem = lab / f"{sid}_{s.bcrx.removesuffix('.bcrx')}"
    sheet = stem.with_suffix(".xlsx")
    if args.rewrite_sheets or not sheet.exists():
        old = read_sheet(sheet) if sheet.exists() else None
        prev = {int(r.id): (r.self_report, r.comment) for r in old.itertuples() if r.self_report or r.comment} if old is not None else {}
        write_sheet(dialog, sheet, f"Разметка лжи — запись {s.video}", prev)
    stem.with_suffix(".srt").write_text(to_srt(dialog, ROLE_LABELS), encoding="utf-8")
    if not stem.with_suffix(".mp4").exists():
        os.link(interim_dir(sid) / "proxy.mp4", stem.with_suffix(".mp4"))
    print(sid, flush=True)
