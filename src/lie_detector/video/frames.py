import json
import subprocess

import numpy as np


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate", "-of", "json", str(path)], capture_output=True, check=True)
    st = json.loads(out.stdout)["streams"][0]
    num, den = map(int, st["r_frame_rate"].split("/"))
    return {"width": st["width"], "height": st["height"], "fps": num / den}


def iter_frames(path, width, height):
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "bgr24",
                             "pipe:1"], stdout=subprocess.PIPE, bufsize=0)
    try:
        yield from read_raw_frames(proc.stdout, width, height)
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def read_raw_frames(stream, width, height):
    size = width * height * 3
    buf = bytearray(size)
    view = memoryview(buf)
    while True:
        got = 0
        while got < size:
            n = stream.readinto(view[got:])
            if not n:
                return
            got += n
        yield np.frombuffer(buf, np.uint8).reshape(height, width, 3).copy()
