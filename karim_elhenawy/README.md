# Karim El Henawy: landmark recognition by matching

Uses Karim Elnazir's SIFT matcher (`shared/matcher.py`) to recognize which of the 6 landmarks
is in each test photo, with KNN-style voting, and evaluates the result.

Run from the project root, in this order (needs `opencv-python numpy matplotlib`):

    python karim_elhenawy/dataset_split.py        # train/test + 5 folds per landmark
    python karim_elhenawy/recognition.py          # all match scores (~4 min on 4 cores, resumable)
    python karim_elhenawy/knn_classifier.py       # choose k and voting rule on the training folds
    python karim_elhenawy/unknown_detection.py    # optional bonus: unknown-place threshold
    python karim_elhenawy/evaluate.py             # final test predictions and metrics

## Split
- Uses the processed images in `dataset/processed/` (no extra preprocessing).
- Test: photos `_1` to `_5` of each landmark = **30 photos**.
- Training/reference gallery: every other photo (`_6` and up, including `_21`, as in
  `shared/config.py`) = **91 photos** (landmarks 1 and 5 have 14, landmark 2 has 15, landmarks 3, 4, 6 have 16).
- Test photos are never in any gallery and are never used for tuning.

## K-fold split (for tuning)
The training photos of **each landmark separately** are shuffled (seed 42) and dealt into 5 folds,
so every fold has about 3 photos of every landmark (fold sizes 19/18/18/18/18).
In round f, fold f is the validation set and the other 4 folds are the gallery.
Saved in `results/karim_elhenawy/splits.csv` (test photos have fold = -1).

## Match score
`num_good` = SIFT matches that pass the 0.75 ratio test and are RANSAC inliers
(`match_images(query, reference).num_good`). All scores are in `results/karim_elhenawy/scores.csv`.

## KNN voting
1. Rank gallery photos by `num_good` (ties: `num_ratio`, then file name).
2. Take the top `k`.
3. Vote by landmark. `majority`: 1 vote each. `weighted`: `num_good` votes each.
4. Tie between landmarks: larger total `num_good` among the top k wins, then the landmark of the highest-ranked neighbour.

## Chosen settings (5-fold cross-validation on training photos only)
k in {1, 3, 5, 7, 9, 11, 15} x {majority, weighted} were tried (`cv_results.csv`).
Best: **k = 5, weighted voting**, CV accuracy **0.934** (85/91). Ties in CV accuracy would go to the smaller k.

## Test results (`accuracy.txt`, `predictions.csv`, `confusion_matrix.png`)
- **Accuracy: 0.833 (25/30)**, macro F1 0.836. Best single match only (1-NN): 0.867.
- Landmarks 3 and 4: 5/5. Landmarks 2 and 6: 4/5. Landmark 5: 5/5. Landmark 1: 2/5.
- All 5 mistakes were predicted as landmark 5 (three of them are landmark 1 photos). Landmark 5 photos
  seem to share many SIFT matches with other scenes, so it attracts wrong votes.
- The test score is below the CV score; with only 5 test photos per landmark, one photo changes
  that landmark's recall by 0.2.

## Unknown place (optional bonus, `unknown_detection.csv`, `unknown_metrics.txt`)
- Confidence = `num_good` of the best match. If it is below the threshold, the answer is "unknown place".
- There are no photos of other places, so an unknown place is simulated by removing the photo's own
  landmark from the gallery.
- Threshold chosen on the training folds only (best balanced accuracy): **52**.
- Test: 28/30 simulated unknown photos rejected, but only 19/30 known photos accepted
  (balanced accuracy 0.783). Many correct matches have fewer than 52 inliers, so this rule
  rejects too many real landmarks.

## Values for the team comparison table
| Method | Test accuracy | Macro F1 | Split | Settings |
|---|---|---|---|---|
| SIFT + RANSAC matching, KNN vote | 0.833 (25/30) | 0.836 | test `_1`-`_5`, gallery `_6`+ (91) | k = 5, weighted |
