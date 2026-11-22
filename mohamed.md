# Mohamed Elsayed — Global Features, SVM, Neural Network, and Panorama

## Responsibility

Extract a global feature representation from each image, train and evaluate an SVM and a small neural network for landmark classification, and build a panorama from overlapping photos. Use Karim Elnazir's matcher for panorama correspondence information.

## Project folder structure

```text
Landmark_Recognition/
├── dataset/
│   └── processed/
│       ├── landmark_1/
│       ├── landmark_2/
│       ├── landmark_3/
│       ├── landmark_4/
│       ├── landmark_5/
│       └── landmark_6/
│
├── shared/
│   ├── matcher.py              # implemented by Karim Elnazir
│   ├── dataset_loader.py       # optional shared image/label loader
│   └── config.py
│
├── mohamed/
│   ├── dataset_split.py
│   ├── feature_extraction.py
│   ├── svm_classifier.py
│   ├── neural_network.py
│   ├── panorama.py
│   ├── seam_evaluation.py
│   └── README.md
│
└── results/
    └── mohamed/
        ├── features/
        ├── svm/
        │   ├── confusion_matrix.png
        │   └── metrics.txt
        ├── neural_network/
        │   ├── confusion_matrix.png
        │   └── metrics.txt
        └── panorama/
            ├── panorama.jpg
            └── seam_check.png
```

This work area uses the processed images created by `karim_elnazir/preprocess.py`. All classification experiments must use the same fixed training/testing split as the rest of the team.

## Files and what each one does

| File | Purpose |
|---|---|
| `dataset_split.py` | Assigns images to training or testing from the filename suffix: `_1` to `_5` are test and `_6` to `_20` are train, for every landmark. |
| `feature_extraction.py` | Converts each image into a fixed-length numeric feature vector. Implement the selected representation, such as a color histogram or a resized grayscale image (optionally reduced using PCA). Fit any learned transformation, including PCA, on training images only, then apply it to test images. |
| `svm_classifier.py` | Trains an SVM using the training feature vectors and labels, predicts the test labels, and saves the evaluation results. |
| `neural_network.py` | Trains a small neural network using the same feature vectors and training labels, predicts the test labels, and saves its evaluation results. |
| `panorama.py` | Selects photos with overlapping views, obtains feature correspondences using Karim Elnazir's shared matcher, estimates image alignment, warps images, and blends them into a panorama. |
| `seam_evaluation.py` | Displays or measures visible seams, misalignment, ghosting, and blending issues in the generated panorama. |

## Dataset and split

The filename format is `<landmark-number>_<photo-number>`, for example `1_1.jpg` or `6_20.jpg`.

For each landmark:

- `_1` through `_5`: testing images.
- `_6` through `_20`: training images.

With 20 photos for each of six landmarks, this produces 90 training images and 30 testing images. Use the same split as Karim El Henawy.

For classification, do not fit PCA, feature scaling, feature selection, or other data-dependent transformations using the test images. Fit them on the training portion only, then transform the test images with the fitted settings.

## Panorama note

Panorama stitching is a separate task from landmark classification. It requires photos of the **same scene with overlapping fields of view**, ideally captured as a sequence while rotating the camera. Random photos of a landmark may not overlap enough to stitch successfully.

The panorama module needs more than a good-match count. Coordinate with Karim Elnazir so `shared/matcher.py` can return matched keypoint coordinates and RANSAC inliers (and, if provided by the matcher, a homography or enough information to estimate one). Use suitable overlapping photos for the panorama; do not assume the classification train/test split automatically provides a panorama sequence.

## Expected outputs

- SVM test accuracy and confusion matrix.
- Neural-network test accuracy and confusion matrix.
- Saved feature representation/settings so experiments can be reproduced.
- A panorama image made from overlapping photos.
- A seam/misalignment check image or short evaluation.
- Accuracy results and metric definitions supplied to Karim El Henawy for the final comparison table.
