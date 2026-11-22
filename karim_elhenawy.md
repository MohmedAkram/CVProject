# Karim El Henawy — Landmark Recognition by Matching

## Responsibility

Use Karim Elnazir's SIFT matcher to recognize the landmark in each test photo. Compare each test image with the training/reference images, predict the landmark with a KNN-style voting method, and evaluate the predictions.

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
│   ├── dataset_loader.py       # optional helper for loading images and labels
│   └── config.py
│
├── karim_elhenawy/
│   ├── dataset_split.py
│   ├── recognition.py
│   ├── knn_classifier.py
│   ├── unknown_detection.py
│   ├── evaluate.py
│   └── README.md
│
└── results/
    └── karim_elhenawy/
        ├── predictions.csv
        ├── confusion_matrix.png
        ├── accuracy.txt
        └── unknown_detection.csv
```

This work area uses the processed images created by `karim_elnazir/preprocess.py`. It should not create a separate, differently preprocessed copy of the dataset.

## Files and what each one does

| File | Purpose |
|---|---|
| `dataset_split.py` | Reads the image names and assigns each image to training or testing using the agreed rule: photo numbers `_1` to `_5` are test; `_6` to `_20` are train, separately for each landmark. |
| `recognition.py` | Loads each test image and compares it against all training/reference images using the shared SIFT matcher. Stores the match score for every query-reference pair. |
| `knn_classifier.py` | Uses the match scores to predict the landmark. For a KNN-style approach, selects the top `k` reference images and votes by landmark, with a documented rule for ties. |
| `unknown_detection.py` | Optional bonus: predicts `unknown place` if the best match score (or another agreed confidence measure) is below a threshold. The threshold must be chosen using training/validation data, not by looking at test outcomes. |
| `evaluate.py` | Compares predictions with the true landmark labels, calculates accuracy, and creates the six-class confusion matrix. If unknown detection is implemented, also reports its results separately. |

## Dataset and split

The filename format is `<landmark-number>_<photo-number>`, for example `1_1.jpg` or `4_17.jpg`.

For each of the six landmarks:

- `_1` through `_5`: testing images.
- `_6` through `_20`: training/reference images.

Expected totals when there are 20 photos per landmark:

- Training/reference: 15 × 6 = 90 images.
- Testing: 5 × 6 = 30 images.

Use only the 90 training/reference images as the gallery for recognition. Each of the 30 test images is queried against that gallery. Do not include a test image in its own reference gallery.

## How the recognition process works

1. Load one test image.
2. Compare it with every training/reference image using `shared/matcher.py`.
3. Record the number of good matches for each reference image.
4. Rank the reference images by match score.
5. Select the top `k` matches and vote for the landmark label.
6. Save the predicted label and scores.
7. Repeat for every test image.
8. Compare predictions with the true labels and calculate the metrics.

The team should decide and document `k`, the voting rule, and how ties are handled. Any tuning must use training data or a validation subset taken only from the training portion.

## Expected outputs

- `predictions.csv`: test filename, true landmark, predicted landmark, and useful match-score information.
- `accuracy.txt`: overall test accuracy and any additional agreed metrics.
- `confusion_matrix.png`: confusion matrix for the six landmarks.
- `unknown_detection.csv` and related metrics if the optional unknown-place feature is implemented.
- Final comparison-table values shared with the team. Provide the metric definition and test split used so results can be compared fairly.
