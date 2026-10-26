import numpy as np

from lie_detector.cli import session_args
from lie_detector.config import interim_dir, processed_dir
from lie_detector.video.frames import iter_frames, probe
from lie_detector.video.pose import RespondentPose

args, todo = session_args()
for sid in todo:
    out, src = processed_dir(sid) / "pose.npz", interim_dir(sid) / "proxy.mp4"
    if out.exists() and not args.force:
        continue
    info, tracker = probe(src), RespondentPose()
    kps, scs, boxes = zip(*(tracker(f) for f in iter_frames(src, info["width"], info["height"])))
    np.savez_compressed(out, kpts=np.stack(kps), scores=np.stack(scs), boxes=np.stack(boxes), **info)
    print(sid, flush=True)
