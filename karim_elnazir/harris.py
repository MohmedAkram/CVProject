import cv2
import numpy as np


def detect_harris(img, max_points=1000, block=2, ksize=3, k=0.04):
    gray = np.float32(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
    response = cv2.cornerHarris(gray, block, ksize, k)
    # keep only local maxima above a fraction of the strongest corner
    local_max = response == cv2.dilate(response, np.ones((5, 5), np.uint8))
    keep = local_max & (response > 0.01 * response.max())
    ys, xs = np.nonzero(keep)
    order = np.argsort(-response[ys, xs])[:max_points]
    return np.stack([xs[order], ys[order]], axis=1).astype(np.float32)


def draw_points(img, points, color=(0, 0, 255)):
    out = img.copy()
    for x, y in points:
        cv2.circle(out, (int(x), int(y)), 3, color, 1)
    return out
