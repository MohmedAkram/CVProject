import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
from shared import config
from karim_elnazir.harris import detect_harris, draw_points
from karim_elnazir.sift import detect_sift, keypoint_xy, draw_keypoints
from karim_elnazir.matcher import match_images

MAX_POINTS = 1000
GRID = 8
TOLERANCE = 3.0
MIN_INLIERS = 15
PHOTOS_PER_LANDMARK = 3


def coverage(points, shape):
    h, w = shape[:2]
    cols = np.clip((points[:, 0] * GRID // w).astype(int), 0, GRID - 1)
    rows = np.clip((points[:, 1] * GRID // h).astype(int), 0, GRID - 1)
    return len(set(zip(rows, cols))) / (GRID * GRID)


def repeatability(pts_a, pts_b, H, shape_b):
    if len(pts_a) == 0 or len(pts_b) == 0:
        return 0.0
    proj = cv2.perspectiveTransform(pts_a.reshape(-1, 1, 2), H).reshape(-1, 2)
    h, w = shape_b[:2]
    inside = (proj[:, 0] >= 0) & (proj[:, 0] < w) & (proj[:, 1] >= 0) & (proj[:, 1] < h)
    proj = proj[inside]
    if len(proj) == 0:
        return 0.0
    dist = np.linalg.norm(proj[:, None, :] - pts_b[None, :, :], axis=2).min(axis=1)
    return float((dist < TOLERANCE).mean())


def detect_both(img):
    harris = detect_harris(img, MAX_POINTS)
    sift = keypoint_xy(detect_sift(img, MAX_POINTS)[0])
    return harris, sift


def save_examples():
    out = config.RESULTS_DIR / "keypoint_images"
    out.mkdir(parents=True, exist_ok=True)
    for path, landmark, photo in config.list_processed("train"):
        if photo != 6:
            continue
        img = cv2.imread(str(path))
        harris = detect_harris(img, MAX_POINTS)
        sift_kp, _ = detect_sift(img, MAX_POINTS)
        cv2.imwrite(str(out / f"{landmark}_{photo}_harris.jpg"), draw_points(img, harris))
        cv2.imwrite(str(out / f"{landmark}_{photo}_sift.jpg"), draw_keypoints(img, sift_kp))


def main():
    # only training images are used here
    train = config.list_processed("train")
    pair_rows = []
    skipped = 0
    for landmark in config.LANDMARKS:
        paths = [p for p, l, _ in train if l == landmark][:PHOTOS_PER_LANDMARK]
        imgs = [cv2.imread(str(p)) for p in paths]
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                result = match_images(paths[i], paths[j])
                if result.homography is None or result.num_good < MIN_INLIERS:
                    skipped += 1
                    continue
                h_a, s_a = detect_both(imgs[i])
                h_b, s_b = detect_both(imgs[j])
                H = result.homography
                pair_rows.append({
                    "landmark": landmark,
                    "image_a": paths[i].name,
                    "image_b": paths[j].name,
                    "sift_inliers": result.num_good,
                    "harris_keypoints": (len(h_a) + len(h_b)) / 2,
                    "sift_keypoints": (len(s_a) + len(s_b)) / 2,
                    "harris_coverage": (coverage(h_a, imgs[i].shape) + coverage(h_b, imgs[j].shape)) / 2,
                    "sift_coverage": (coverage(s_a, imgs[i].shape) + coverage(s_b, imgs[j].shape)) / 2,
                    "harris_repeatability": repeatability(h_a, h_b, H, imgs[j].shape),
                    "sift_repeatability": repeatability(s_a, s_b, H, imgs[j].shape),
                })

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if pair_rows:
        with open(config.RESULTS_DIR / "detector_comparison_pairs.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(pair_rows[0].keys()))
            writer.writeheader()
            writer.writerows(pair_rows)

    summary = []
    for name in ["harris", "sift"]:
        summary.append([
            name,
            round(float(np.mean([r[f"{name}_keypoints"] for r in pair_rows])), 1),
            round(float(np.mean([r[f"{name}_coverage"] for r in pair_rows])), 3),
            round(float(np.mean([r[f"{name}_repeatability"] for r in pair_rows])), 3),
            len(pair_rows),
        ])
    with open(config.RESULTS_DIR / "detector_comparison.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["detector", "mean_keypoints", "mean_grid_coverage", "mean_repeatability", "pairs_used"])
        writer.writerows(summary)

    save_examples()
    print("pairs used:", len(pair_rows), "skipped (too few inliers):", skipped)
    for row in summary:
        print(row)


if __name__ == "__main__":
    main()
