# -*- coding: utf-8 -*-
"""
ExACT AFE (Adaptive Frame Events): motion-driven recursive temporal splitting + white-bg render
+ morphological denoise. Faithfully ported from ExACT 'Dataloader/AFE Preprocessing'. DAVIS346.
ev = [N,4] = (x,y,t,p) sorted by t -> variable-length list of [S,S,3] uint8 frames. New file.
"""
import numpy as np
import cv2

W0, H0 = 346, 260
SAMPLE_MIN = 100000
SAMPLE_THRESH = 50


def _counts(ev):
    x = ev[:, 0].astype(np.int64); y = ev[:, 1].astype(np.int64); p = ev[:, 3].astype(np.int64)
    pos = np.zeros(H0 * W0, np.float32); neg = np.zeros(H0 * W0, np.float32)
    np.add.at(pos, x[p == 1] + W0 * y[p == 1], 1.0)
    np.add.at(neg, x[p == 0] + W0 * y[p == 0], 1.0)
    return pos.reshape(H0, W0), neg.reshape(H0, W0)


def _event_image(ev, S=224):                              # white bg, ON->red OFF->blue (ExACT rgb)
    pos, neg = _counts(ev)
    fr = 255 * (1 - (pos[..., None] * np.array([0, 255, 255], np.float32)
                     + neg[..., None] * np.array([255, 255, 0], np.float32)) / 255)
    return cv2.resize(np.clip(fr, 0, 255).astype(np.uint8), (S, S))


def _abs_image(ev):                                       # denoised count map (for motion test)
    pos, neg = _counts(ev)
    k = np.ones((2, 2), np.uint8)
    pos = cv2.dilate(cv2.erode(pos, k), k); neg = cv2.dilate(cv2.erode(neg, k), k)
    return pos + neg


def _sufficient(ev, thresh=SAMPLE_THRESH):
    N = len(ev); h = N // 2
    if h < 1:
        return True
    d = np.abs(_abs_image(ev[:h]) - _abs_image(ev[h:]))
    return 200.0 * (d.sum() / max(N, 1)) <= thresh


def adaptive_sample(ev, S=224, min_n=SAMPLE_MIN, thresh=SAMPLE_THRESH):
    N = len(ev); half = N // 2
    if _sufficient(ev, thresh):                           # static enough -> single frame
        return [_event_image(ev, S)]
    if half <= min_n:                                     # leaf -> two half-frames
        return [_event_image(ev[:half], S), _event_image(ev[half:], S)]
    return adaptive_sample(ev[:half], S, min_n, thresh) + adaptive_sample(ev[half:], S, min_n, thresh)
