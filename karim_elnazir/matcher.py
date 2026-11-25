import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from shared import config
from karim_elnazir.sift import detect_sift


@dataclass
class Features:
    points: np.ndarray        # (N, 2) keypoint x, y
    descriptors: np.ndarray   # (N, 128)


@dataclass
class MatchResult:
    num_ratio: int            # matches that passed the ratio test
    num_good: int             # matches that also survived RANSAC (the score)
    pts_a: np.ndarray         # (num_ratio, 2) points in image a
    pts_b: np.ndarray         # (num_ratio, 2) matching points in image b
    inlier_mask: np.ndarray   # (num_ratio,) True for RANSAC inliers
    homography: np.ndarray    # 3x3 matrix mapping a to b, or None


def load_image(image):
    if isinstance(image, (str, Path)):
        img = cv2.imread(str(image))
        if img is None:
            raise FileNotFoundError(image)
        return img
    return image


def extract_features(image):
    keypoints, descriptors = detect_sift(load_image(image))
    if descriptors is None:
        return Features(np.zeros((0, 2), np.float32), np.zeros((0, 128), np.float32))
    return Features(np.float32([k.pt for k in keypoints]), descriptors)


def _as_features(x):
    return x if isinstance(x, Features) else extract_features(x)


def match_images(a, b, ratio=config.RATIO, ransac_thresh=config.RANSAC_THRESH):
    """a and b can be image paths, image arrays, or Features from extract_features."""
    fa = _as_features(a)
    fb = _as_features(b)
    empty = np.zeros((0, 2), np.float32)
    if len(fa.descriptors) < 2 or len(fb.descriptors) < 2:
        return MatchResult(0, 0, empty, empty, np.zeros(0, bool), None)

    pairs = cv2.BFMatcher(cv2.NORM_L2).knnMatch(fa.descriptors, fb.descriptors, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance]

    pts_a = np.float32([fa.points[m.queryIdx] for m in good]).reshape(-1, 2)
    pts_b = np.float32([fb.points[m.trainIdx] for m in good]).reshape(-1, 2)
    mask = np.zeros(len(good), bool)
    H = None
    if len(good) >= config.MIN_MATCHES:
        H, m = cv2.findHomography(pts_a, pts_b, cv2.RANSAC, ransac_thresh)
        if H is not None:
            mask = m.ravel().astype(bool)
    return MatchResult(len(good), int(mask.sum()), pts_a, pts_b, mask, H)


def count_good_matches(a, b):
    return match_images(a, b).num_good


def draw_matches(img_a, img_b, result, path=None):
    h = max(img_a.shape[0], img_b.shape[0])
    canvas = np.zeros((h, img_a.shape[1] + img_b.shape[1], 3), np.uint8)
    canvas[:img_a.shape[0], :img_a.shape[1]] = img_a
    canvas[:img_b.shape[0], img_a.shape[1]:] = img_b
    offset = img_a.shape[1]
    for pa, pb, ok in zip(result.pts_a, result.pts_b, result.inlier_mask):
        if not ok:
            continue
        p1 = (int(pa[0]), int(pa[1]))
        p2 = (int(pb[0]) + offset, int(pb[1]))
        cv2.line(canvas, p1, p2, (0, 255, 0), 1)
        cv2.circle(canvas, p1, 3, (0, 255, 0), 1)
        cv2.circle(canvas, p2, 3, (0, 255, 0), 1)
    if path is not None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), canvas)
    return canvas


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python karim_elnazir/matcher.py image_a image_b")
        sys.exit(1)
    img_a = load_image(sys.argv[1])
    img_b = load_image(sys.argv[2])
    result = match_images(img_a, img_b)
    print("passed ratio test:", result.num_ratio)
    print("good matches (RANSAC inliers):", result.num_good)
    out = config.RESULTS_DIR / "match_visualizations" / "example_match.jpg"
    draw_matches(img_a, img_b, result, out)
    print("saved", out)
