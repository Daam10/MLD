import subprocess
import threading
import zipfile
from pathlib import Path
from typing import BinaryIO


def open_video(name: str, archive: Path) -> BinaryIO:
    return zipfile.ZipFile(archive).open(name)


def read_bcrx_bytes(name: str, archive: Path) -> bytes:
    zf = zipfile.ZipFile(archive, metadata_encoding="cp866")
    return zf.read(next(i for i in zf.infolist() if i.filename == name or i.filename.endswith("/" + name)))


def pipe_to_ffmpeg(src: BinaryIO, args: list[str], log: Path | None = None) -> int:
    proc = subprocess.Popen(["ffmpeg", "-hide_banner", "-nostdin", "-y", "-f", "mpegts", "-i", "pipe:0", *args],
                            stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    err = []
    reader = threading.Thread(target=lambda: err.append(proc.stderr.read()))
    reader.start()
    try:
        while chunk := src.read(8 << 20):
            proc.stdin.write(chunk)
    except BrokenPipeError:
        pass
    proc.stdin.close()
    code = proc.wait()
    reader.join()
    if log:
        log.write_bytes(err[0] if err else b"")
    return code
