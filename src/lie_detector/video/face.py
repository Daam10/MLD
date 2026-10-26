import cv2
import numpy as np


def roi_polygons(lm):
    brow, eyes = lm[17:27], lm[36:48]
    h = max(np.mean(eyes[:, 1]) - np.mean(brow[:, 1]), 2.0)
    top = brow - [0, 1.2 * h]
    return {"forehead": np.vstack([brow[[0, 2, 4, 5, 7, 9]] - [0, 0.15 * h], top[[9, 7, 5, 4, 2, 0]]]),
            "cheek_l": np.vstack([lm[[47, 46]] + [0, 0.25 * h], lm[[35]], lm[[54]] - [0, 0.2 * h], lm[[12, 13, 14]]]),
            "cheek_r": np.vstack([lm[[41, 40]] + [0, 0.25 * h], lm[[31]], lm[[48]] - [0, 0.2 * h], lm[[4, 3, 2]]])}


def face_box(lm, scale=1.6):
    c, side = (lm.min(0) + lm.max(0)) / 2, (lm.max(0) - lm.min(0)).max() * scale
    return np.array([c[0] - side / 2, c[1] - side / 2, c[0] + side / 2, c[1] + side / 2])


def roi_means(frame, lm, valid_face):
    means, counts = np.full((3, 3), np.nan, np.float32), np.zeros(3, np.int32)
    if not valid_face:
        return means, counts
    box = face_box(lm, 1.3).astype(int)
    x0, y0, x1, y1 = max(box[0], 0), max(box[1], 0), min(box[2], frame.shape[1]), min(box[3], frame.shape[0])
    if x1 - x0 < 10 or y1 - y0 < 10:
        return means, counts
    crop = frame[y0:y1, x0:x1]
    ycc = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
    skin = (ycc[..., 1] >= 133) & (ycc[..., 1] <= 173) & (ycc[..., 2] >= 77) & (ycc[..., 2] <= 127)
    for k, poly in enumerate(roi_polygons(lm).values()):
        mask = np.zeros(crop.shape[:2], np.uint8)
        cv2.fillPoly(mask, [np.round(poly - [x0, y0]).astype(np.int32)], 1)
        m = (mask > 0) & skin
        if m.sum() >= 20:
            means[k], counts[k] = crop[m][:, ::-1].mean(0), int(m.sum())
    return means, counts


def face_crop(frame, box, size=72):
    s = size / (box[2] - box[0])
    return cv2.warpAffine(frame, np.float32([[s, 0, -box[0] * s], [0, s, -box[1] * s]]), (size, size),
                          flags=cv2.INTER_AREA, borderMode=cv2.BORDER_REPLICATE)


def smooth_landmarks(face_kpts, face_scores, thr=0.3, alpha=0.5):
    out, valid, prev = np.empty_like(face_kpts), face_scores.mean(1) > thr, None
    for t in range(len(face_kpts)):
        cur = face_kpts[t] if valid[t] else prev
        if cur is None:
            out[t] = np.nan
            continue
        prev = cur if prev is None else alpha * cur + (1 - alpha) * prev
        out[t] = prev
    return out, valid
