import io
import re
import zipfile
import zlib
from dataclasses import dataclass
from xml.etree import ElementTree as ET

import numpy as np

DEVICE_NAMES = {"Cardio": "ecg", "Breath": "resp", "Skin": "gsr", "ECG": "ecg", "RIP": "resp", "GSR": "gsr",
                "PPG pulse": "ppg", "Heart Rate pulse": "hr_oximeter", "SpO2 pulse": "spo2", "Flow Rate temp": "airflow"}


@dataclass
class BioRecording:
    fs: float
    signals: dict[str, np.ndarray]
    fs_of: dict[str, float]
    duration_s: float
    events_s: np.ndarray


def _tag(e: ET.Element) -> str:
    return re.sub(r"\{.*?\}", "", e.tag)


def _text(root: ET.Element, name: str) -> str | None:
    return next((e.text for e in root.iter() if _tag(e) == name), None)


def read_bcrx(data: bytes, rename: dict[str, str] | None = None) -> BioRecording:
    zf = zipfile.ZipFile(io.BytesIO(data))
    root = ET.fromstring(zf.read("header.xml"))
    sigs = [next(e for e in d if _tag(e) == "Signal") for d in root.iter() if _tag(d) == "CustomSignalDefinition"]
    names = [_text(s, "Name") for s in sigs]
    rn = lambda n: (rename or {}).get(n, n)
    signals = {}
    for ch, name in enumerate(names):
        parts = sorted((n for n in zf.namelist() if re.fullmatch(rf"{ch}_\d+\.rec", n)), key=lambda n: int(n[2:-4].split("_")[-1]))
        signals[rn(name)] = np.frombuffer(b"".join(zlib.decompress(zf.read(p), -15) for p in parts), "<f4").copy()
    fs_of = {rn(n): float(_text(s, "SamplesPerSecond")) for n, s in zip(names, sigs)}
    events = np.array([int(e.text) / 1e7 for e in root.iter() if _tag(e) == "RecordedTime"])
    return BioRecording(fs_of[rn(names[0])], signals, fs_of, int(_text(root, "RecordingLengthTicks")) / 1e7, events)


def load_session_bio(session) -> BioRecording:
    from lie_detector.io.raw import read_bcrx_bytes
    return read_bcrx(read_bcrx_bytes(session.bcrx, session.bio_zip), rename=DEVICE_NAMES)
