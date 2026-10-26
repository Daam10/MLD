from __future__ import annotations

import av
import cv2
import numpy as np

from lie_detector.video.face import roi_polygons

BODY_LINKS = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
              (0, 1), (0, 2), (1, 3), (2, 4), (15, 17), (15, 18), (15, 19), (16, 20), (16, 21), (16, 22)]
HAND_LINKS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
              (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
ROI_COLORS = {"forehead": (0, 200, 255), "cheek_l": (255, 120, 0), "cheek_r": (255, 120, 0)}
pt = lambda p: tuple(np.int32(p))


class VideoReader:
    def __init__(self, path):
        self.container = av.open(path)
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        self.fps, self.tb = float(self.stream.average_rate), float(self.stream.time_base)
        self.n_frames = self.stream.frames or int(self.container.duration / 1e6 * self.fps)
        self._iter, self._last_idx, self._last = None, -10, None

    def frame(self, idx):
        idx = int(np.clip(idx, 0, max(self.n_frames - 1, 0)))
        if idx == self._last_idx:
            return self._last
        if self._iter is None or not 0 < idx - self._last_idx <= 12:
            self.container.seek(int(idx / self.fps / self.tb), stream=self.stream, backward=True, any_frame=False)
            self._iter, self._last_idx = self.container.decode(self.stream), -10
        for fr in self._iter:
            i = int(round(fr.pts * self.tb * self.fps))
            if i >= idx:
                self._last_idx, self._last = i, fr.to_ndarray(format="rgb24")
                break
        return self._last


def draw_overlays(img, kp, sc, show_pose=True, show_face=True, thr=0.55):
    if kp is None:
        return img
    out = img.copy()
    if show_pose:
        links = [(a, b, (0, 230, 118), 2) for a, b in BODY_LINKS] + [(o + a, o + b, (255, 64, 129), 1) for o in (91, 112) for a, b in HAND_LINKS]
        for a, b, col, w in links:
            if sc[a] > thr and sc[b] > thr:
                cv2.line(out, pt(kp[a]), pt(kp[b]), col, w, cv2.LINE_AA)
        for i in np.flatnonzero(sc[:17] > thr):
            cv2.circle(out, pt(kp[i]), 3, (255, 255, 255), -1, cv2.LINE_AA)
    if show_face and sc[23:91].mean() > 0.3:
        for name, poly in roi_polygons(kp[23:91]).items():
            cv2.polylines(out, [np.int32(poly)], True, ROI_COLORS[name], 1, cv2.LINE_AA)
    return out


def face_zoom(img, kp, sc, size=320):
    if kp is None or sc[23:91].mean() < 0.3:
        return None
    lm = kp[23:91]
    side = max((lm.max(0) - lm.min(0)).max() * 1.8, 20)
    x0, y0 = (lm.min(0) + lm.max(0)) / 2 - side / 2
    k = size / side
    return cv2.warpAffine(img, np.float32([[k, 0, -x0 * k], [0, k, -y0 * k]]), (size, size), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
