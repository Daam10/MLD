import subprocess
import threading

import numpy as np

from lie_detector.cli import session_args
from lie_detector.config import interim_dir, processed_dir
from lie_detector.io.raw import open_video
from lie_detector.video.face import face_box, face_crop, roi_means, smooth_landmarks
from lie_detector.video.frames import read_raw_frames

W, H = 1920, 1080


def stream_frames(video, archive):
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-nostdin", "-f", "mpegts", "-i", "pipe:0", "-map", "0:v:0", "-vf", "yadif=0",
                             "-r", "25", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)

    def feed():
        with open_video(video, archive) as f:
            try:
                while chunk := f.read(8 << 20):
                    proc.stdin.write(chunk)
            except OSError:
                pass
        proc.stdin.close()

    threading.Thread(target=feed, daemon=True).start()
    yield from read_raw_frames(proc.stdout, W, H)
    proc.wait()


def background_mean(frame, box):
    side = box[2] - box[0]
    for dx in (1.3, -1.3):
        x0, y0 = int(box[0] + dx * side), int(box[1] - 0.3 * side)
        x1, y1 = x0 + int(0.6 * side), y0 + int(0.6 * side)
        if x0 >= 0 and y0 >= 0 and x1 <= frame.shape[1] and y1 <= frame.shape[0]:
            return frame[y0:y1, x0:x1].reshape(-1, 3).mean(0)[::-1]
    return np.full(3, np.nan, np.float32)


args, todo = session_args()
for sid, s in todo.items():
    out = processed_dir(sid) / "face_rgb.npz"
    if out.exists() and not args.force:
        continue
    pose = np.load(processed_dir(sid) / "pose.npz")
    lm_all, valid = smooth_landmarks(pose["kpts"][:, 23:91] * W / float(pose["width"]), pose["scores"][:, 23:91])
    T = len(lm_all)
    rgb, counts = np.full((T, 3, 3), np.nan, np.float32), np.zeros((T, 3), np.int32)
    boxes, bg = np.full((T, 4), np.nan, np.float32), np.full((T, 3), np.nan, np.float32)
    crops = np.lib.format.open_memmap(interim_dir(sid) / "face_crops.npy", mode="w+", dtype=np.uint8, shape=(T, 72, 72, 3))
    for t, frame in zip(range(T), stream_frames(s.video, s.video_zip)):
        lm = lm_all[t]
        rgb[t], counts[t] = roi_means(frame, lm, bool(valid[t]) and np.isfinite(lm).all())
        if np.isfinite(lm).all():
            boxes[t] = face_box(lm)
            crops[t], bg[t] = face_crop(frame, boxes[t]), background_mean(frame, boxes[t])
    crops.flush()
    np.savez_compressed(out, rgb=rgb, counts=counts, valid=valid, boxes=boxes, bg=bg, fps=25.0)
    print(sid, flush=True)
