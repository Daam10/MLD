import numpy as np

from lie_detector.cli import session_args
from lie_detector.config import interim_dir, processed_dir
from lie_detector.video.breathing import chest_roi, flow_breathing
from lie_detector.video.frames import probe

args, todo = session_args()
for sid in todo:
    out, src = processed_dir(sid) / "breath_video.npz", interim_dir(sid) / "proxy.mp4"
    if out.exists() and not args.force:
        continue
    p, info = np.load(processed_dir(sid) / "pose.npz"), probe(src)
    np.savez_compressed(out, **flow_breathing(src, info["width"], info["height"], chest_roi(p["kpts"], p["scores"]), info["fps"]))
    print(sid, flush=True)
