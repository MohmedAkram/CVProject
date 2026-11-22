# Karim Elnazir — Feature Detection and Image Matching

## Responsibility

Prepare the images, compare Harris and SIFT keypoint detection, implement SIFT matching with the ratio test and RANSAC, and provide a reusable matcher for the other two team members.

## Project folder structure

```text
Landmark_Recognition/
├── dataset/
│   ├── original/
│   │   ├── landmark_1/        # 1_1, 1_2, ... (original files)
│   │   ├── landmark_2/        # 2_1, 2_2, ...
│   │   ├── landmark_3/
│   │   ├── landmark_4/
│   │   ├── landmark_5/
│   │   └── landmark_6/
│   └── processed/             # generated copies of every original image
│       ├── landmark_1/
│       ├── landmark_2/
│       ├── landmark_3/
│       ├── landmark_4/
│       ├── landmark_5/
│       └── landmark_6/
│
├── karim_elnazir/
│   ├── preprocess.py
│   ├── harris.py
│   ├── sift.py
│   ├── matcher.py
│   ├── compare_detectors.py
│   ├── plot_results.py
│   └── README.md
│
├── shared/
│   ├── matcher.py              # stable interface used by both teammates
│   └── config.py               # shared paths and image/landmark settings
│
└── results/
    └── karim_elnazir/
        ├── keypoint_images/
        ├── match_visualizations/
        ├── plots/
        └── detector_comparison.csv
```

The `dataset/original/` folders contain the collected photos. Do not overwrite or modify these originals. Preprocessing creates standardized copies in `dataset/processed/`.

## Files and what each one does

| File | Purpose |
|---|---|
| `preprocess.py` | Reads every image from all six original landmark folders, applies the agreed resizing/format/color preprocessing, and saves a processed copy while preserving its landmark and filename. It processes both eventual training and testing images; it does not use test images to fit or tune anything. |
| `harris.py` | Detects Harris corners/keypoints in an image and provides the keypoint locations and/or a visualization. |
| `sift.py` | Detects SIFT keypoints and computes their descriptors for an image. |
| `matcher.py` | Takes two images, detects SIFT features, matches descriptors, applies Lowe's ratio test, then uses RANSAC to reject geometrically inconsistent matches. Returns the good-match count and the matching information needed by the other modules. |
| `compare_detectors.py` | Runs Harris and SIFT on selected images and compares their repeatability/number and distribution of detected keypoints under different viewpoints or lighting conditions, where those conditions can be identified. |
| `plot_results.py` | Creates the required plots, including good matches versus viewing angle and good matches versus lighting, when the image collection has reliable angle/lighting labels. |
| `shared/matcher.py` | Exposes the agreed public matcher functions so Karim El Henawy and Mohamed can import the same implementation instead of copying code. |

## Dataset naming and split

The first number identifies the landmark; the second number identifies the photo:

- `1_1` to `1_20`: landmark 1
- `2_1` to `2_20`: landmark 2
- Continue the same pattern through landmark 6.

For **each landmark**, photos numbered `_1` through `_5` are testing photos. Photos `_6` through `_20` are training photos. This gives 5 test and 15 training photos per landmark (30 test and 90 training photos overall, assuming 20 photos per landmark).

Keep this split fixed. Apply the same preprocessing procedure to all images, but preserve the split labels and never use testing images to tune thresholds, parameters, or models. The split can be represented in a CSV manifest or generated from the numeric filename suffix.

## Matcher interface required by the team

The matcher must support two uses:

1. **Recognition:** return the number of good SIFT matches between a query image and a reference image.
2. **Panorama stitching:** provide the matched keypoint coordinates and RANSAC inlier information (and, where appropriate, the estimated homography), not just the match count. A count alone is not enough to align images into a panorama.

Agree on function names, input format, and return values with both teammates before they integrate their code. Keep the public interface stable.

## Expected outputs

- Harris-versus-SIFT comparison table.
- Example image(s) showing Harris and SIFT keypoints.
- SIFT matching visualization showing accepted matches/inliers between two images.
- Good-match count returned by the matcher.
- Plot of matches versus angle and plot of matches versus lighting, **only if** those conditions were recorded reliably. Do not invent angle or lighting labels for randomly collected photos.
- Shared matcher module used by Karim El Henawy and Mohamed.
