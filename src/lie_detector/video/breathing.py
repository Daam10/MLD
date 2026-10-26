import cv2
import numpy as np

from lie_detector.signals import bandpass, zscore
from lie_detector.video.frames import iter_frames


def chest_roi(kpts, scores, conf=0.4):
    k = kpts[(scores[:, [5, 6, 11, 12]] > conf).all(1)]
    sh, hip = np.median((k[:, 5] + k[:, 6]) / 2, 0), np.median((k[:, 11] + k[:, 12]) / 2, 0)
    trunk = np.linalg.norm(sh - hip)
    half_w = 0.7 * max(np.median(np.abs(k[:, 5, 0] - k[:, 6, 0])), 0.4 * trunk)
    return int(sh[0] - half_w), int(sh[1] - 0.15 * trunk), int(sh[0] + half_w), int(sh[1] + 0.45 * trunk)


def flow_breathing(video, width, height, roi, fs):
    x0, y0, x1, y1 = max(0, roi[0]), max(0, roi[1]), min(width, roi[2]), min(height, roi[3])
    prev, vy = None, []
    for fr in iter_frames(video, width, height):
        g = cv2.cvtColor(fr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (max(g.shape[1] // 2, 8), max(g.shape[0] // 2, 8)))
        vy.append(0.0 if prev is None else float(np.median(cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 2, 15, 2, 5, 1.1, 0)[..., 1])))
        prev = g
    breath = zscore(bandpass(-np.cumsum(vy), fs, 0.1, 0.6))
    return {"breath": breath.astype(np.float32), "vy": np.asarray(vy, np.float32), "roi": np.array([x0, y0, x1, y1]), "fs": fs}
