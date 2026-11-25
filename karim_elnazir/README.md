# Karim Elnazir: feature detection and matching

Run everything from the project root (the folder that contains `OriginalImages`).

    pip install opencv-python numpy matplotlib
    python karim_elnazir/preprocess.py
    python karim_elnazir/compare_detectors.py
    python karim_elnazir/matcher.py dataset/processed/landmark_1/1_1.jpeg dataset/processed/landmark_1/1_6.jpeg
    python karim_elnazir/plot_results.py

## Preprocessing
Images are read from `OriginalImages/` (never changed) and saved to `dataset/processed/landmark_N/`.
The longest side is resized to 800 px and the aspect ratio is kept. Color is kept; SIFT and Harris
convert to grayscale themselves. `dataset/manifest.csv` lists every image with its split.

## Split
Photos `_1` to `_5` of each landmark are test. All other photos (`_6` and up, including `_21`) are train.
Settings are in `shared/config.py`.

## Matcher (used by the other two)
    from shared.matcher import extract_features, match_images, count_good_matches

- `count_good_matches(a, b)` returns the number of good matches (ratio test 0.75, then RANSAC inliers).
- `match_images(a, b)` returns a `MatchResult` with `num_ratio`, `num_good`, `pts_a`, `pts_b`,
  `inlier_mask` and `homography` (3x3, maps image a to image b, or None).
- `a` and `b` can be file paths, image arrays, or the result of `extract_features(image)`.
  Use `extract_features` once per image to avoid recomputing SIFT for every pair.
- If there are fewer than 8 ratio-test matches, the score is 0 and the homography is None.

## Detector comparison
`compare_detectors.py` uses training images only (first 3 photos of each landmark). For every pair with a
valid homography it compares the number of keypoints, grid coverage and repeatability (share of keypoints
that have a detection within 3 px after the homography). The homography comes from SIFT matches, so the
repeatability numbers favor SIFT-friendly pairs.

## Angle and lighting plots
No angle or lighting labels were recorded. `plot_results.py` only runs if
`results/karim_elnazir/conditions.csv` exists (columns: image_a, image_b, angle_deg, lighting).
