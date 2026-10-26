import numpy as np

from lie_detector.gpu import setup_onnxruntime

BODY = {"nose": 0, "l_ear": 3, "r_ear": 4, "l_shoulder": 5, "r_shoulder": 6, "l_elbow": 7, "r_elbow": 8,
        "l_wrist": 9, "r_wrist": 10, "l_hip": 11, "r_hip": 12, "l_knee": 13, "r_knee": 14, "l_ankle": 15, "r_ankle": 16}
FEET, FACE, L_HAND, R_HAND = slice(17, 23), slice(23, 91), slice(91, 112), slice(112, 133)


def _iou(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    return inter / (area(a) + area(b) - inter + 1e-9)


def _box_from_kpts(kp, sc, shape, thr=0.3, pad=0.15):
    m = sc > thr
    if m.sum() < 8:
        return None
    (x0, y0), (x1, y1) = kp[m].min(0), kp[m].max(0)
    w, h = x1 - x0, y1 - y0
    return np.array([max(0, x0 - pad * w), max(0, y0 - pad * h), min(shape[1], x1 + pad * w), min(shape[0], y1 + pad * h)])


class RespondentPose:
    def __init__(self, mode="performance", det_every=10, device="cuda"):
        setup_onnxruntime()
        from rtmlib import Wholebody
        wb = Wholebody(mode=mode, backend="onnxruntime", device=device)
        self.det, self.pose, self.det_every, self.prev_box, self.i = wb.det_model, wb.pose_model, det_every, None, 0

    def __call__(self, frame):
        box = None
        if self.prev_box is None or self.i % self.det_every == 0:
            boxes = self.det(frame)
            if len(boxes):
                if self.prev_box is None:
                    box = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
                else:
                    box = max(boxes, key=lambda b: _iou(b, self.prev_box))
                    box = box if _iou(box, self.prev_box) >= 0.1 else self.prev_box
        box = self.prev_box if box is None else box
        self.i += 1
        if box is None:
            return np.full((133, 2), np.nan, np.float32), np.zeros(133, np.float32), np.full(4, np.nan)
        kp, sc = self.pose(frame, bboxes=[box])
        kp, sc = kp[0].astype(np.float32), sc[0].astype(np.float32)
        nb = _box_from_kpts(kp, sc, frame.shape)
        self.prev_box = box if nb is None else nb
        return kp, sc, np.asarray(box, np.float32)
