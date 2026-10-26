import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"


def _yaml(p: Path) -> dict:
    return (yaml.safe_load(p.read_text(encoding="utf-8")) or {}) if p.exists() else {}


@lru_cache
def load_config() -> dict:
    return _yaml(CONFIG_DIR / "default.yaml")


def path(key: str) -> Path:
    return (REPO_ROOT / load_config()["paths"][key]).resolve()


def _mkdir(d: Path) -> Path:
    d.mkdir(parents=True, exist_ok=True)
    return d


def interim_dir(*parts: str) -> Path:
    return _mkdir(path("interim_dir").joinpath(*parts))


def processed_dir(*parts: str) -> Path:
    return _mkdir(path("processed_dir").joinpath(*parts))


def hf_token() -> str | None:
    f = path("hf_token_file")
    return os.environ.get("HF_TOKEN") or (f.read_text(encoding="utf-8").strip() if f.exists() else None)


@dataclass(frozen=True)
class Session:
    id: str
    video: str
    scenario: str | None
    sex: str
    age: int
    source: str
    video_zip: Path
    bio_zip: Path
    bcrx: str | None = None
    tab_no: int | None = None

    @property
    def in_study(self) -> bool:
        return bool(self.scenario)

    def condition_of_topic(self) -> dict[str, str]:
        if not self.scenario:
            return {}
        reg = _yaml(CONFIG_DIR / "sessions.yaml")
        return {reg["topics"][c]: cond for c, cond in zip(self.scenario, reg["conditions"])}


@lru_cache
def sessions() -> dict[str, Session]:
    reg = _yaml(CONFIG_DIR / "sessions.yaml")
    local = _yaml(CONFIG_DIR / "subjects.local.yaml").get("subjects", {})
    out = {}
    for sid, s in reg["sessions"].items():
        src, loc = reg["sources"][s["source"]], local.get(sid, {})
        out[sid] = Session(sid, s["video"], s.get("scenario"), s["sex"], s.get("age", reg["defaults"]["age"]),
                           s["source"], (REPO_ROOT / src["video_zip"]).resolve(),
                           (REPO_ROOT / src["bio_zip"]).resolve(), loc.get("bcrx"), loc.get("tab_no"))
    return out
