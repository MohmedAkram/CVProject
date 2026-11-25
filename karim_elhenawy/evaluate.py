import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared import config
from karim_elhenawy.dataset_split import load_splits
from karim_elhenawy.knn_classifier import load_scores, load_params, predict, only

OUT_DIR = config.ROOT / "results" / "karim_elhenawy"
PRED_CSV = OUT_DIR / "predictions.csv"
ACC_TXT = OUT_DIR / "accuracy.txt"
CM_PNG = OUT_DIR / "confusion_matrix.png"
UNKNOWN_TXT = OUT_DIR / "unknown_metrics.txt"
LABELS = list(config.LANDMARKS)


def confusion_matrix(true, pred):
    cm = np.zeros((len(LABELS), len(LABELS)), int)
    for t, p in zip(true, pred):
        cm[LABELS.index(t), LABELS.index(p)] += 1
    return cm


def precision_recall_f1(cm):
    tp = np.diag(cm)
    precision = tp / np.maximum(cm.sum(axis=0), 1)
    recall = tp / np.maximum(cm.sum(axis=1), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    return precision, recall, f1


def plot_cm(cm, title):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues", vmin=0)
    names = [f"L{l}" for l in LABELS]
    ax.set_xticks(range(len(LABELS)), names)
    ax.set_yticks(range(len(LABELS)), names)
    ax.set_xlabel("Predicted landmark")
    ax.set_ylabel("True landmark")
    ax.set_title(title)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(CM_PNG, dpi=150)
    plt.close(fig)


def main():
    params = load_params()
    k, rule = params["k"], params["rule"]
    splits = load_splits()
    scores = load_scores()
    gallery = {r["filename"] for r in splits if r["split"] == "train"}
    test = [r for r in splits if r["split"] == "test"]

    true, pred, nearest, rows = [], [], [], []
    for r in test:
        p, votes, top = predict(only(scores[r["filename"]], gallery), k, rule)
        true.append(r["landmark"])
        pred.append(p)
        nearest.append(top[0][1])
        rows.append([r["filename"], r["landmark"], p, int(p == r["landmark"]),
                     top[0][0], top[0][1], top[0][2],
                     " ".join(f"L{lm}:{votes[lm]:g}" for lm in sorted(votes)),
                     " ".join(f"{name}({good})" for name, _, good, _ in top)])

    with open(PRED_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["test_image", "true_landmark", "predicted_landmark", "correct",
                         "best_reference", "best_reference_landmark", "best_num_good",
                         "votes", f"top_{k}_references(num_good)"])
        writer.writerows(rows)

    cm = confusion_matrix(true, pred)
    precision, recall, f1 = precision_recall_f1(cm)
    acc = np.trace(cm) / len(test)
    nn_acc = np.mean(np.array(nearest) == np.array(true))

    lines = [
        "Landmark recognition by SIFT matching + KNN voting (Karim El Henawy)",
        "",
        f"Test accuracy: {acc:.3f} ({np.trace(cm)}/{len(test)} test photos correct)",
        f"Macro F1:      {f1.mean():.3f}",
        f"1-NN accuracy (best single match only, for reference): {nn_acc:.3f}",
        "",
        "Per landmark:",
        f"{'landmark':>9} {'precision':>10} {'recall':>8} {'F1':>6} {'test photos':>12}",
    ]
    for i, lm in enumerate(LABELS):
        lines.append(f"{lm:>9} {precision[i]:>10.3f} {recall[i]:>8.3f} {f1[i]:>6.3f} {cm[i].sum():>12}")
    lines += [
        "",
        "Settings:",
        f"  k = {k}, voting rule = {rule} (chosen by {params['n_folds']}-fold cross-validation "
        f"on training photos only, CV accuracy {params['cv_accuracy']:.3f})",
        f"  ties: {params['tie_rule']}",
        "  match score = num_good (SIFT, ratio test 0.75, RANSAC inliers), shared/matcher.py",
        "",
        "Split:",
        f"  test = photos _1 to _5 of each landmark ({len(test)} photos)",
        f"  gallery = all other photos, _6 and up incl. _21 ({len(gallery)} photos)",
        "  metric = accuracy = correct test predictions / all test predictions",
    ]
    if UNKNOWN_TXT.exists():
        lines += ["", UNKNOWN_TXT.read_text().rstrip()]
    ACC_TXT.write_text("\n".join(lines) + "\n")

    plot_cm(cm, f"Test confusion matrix (k={k}, {rule}) - accuracy {acc:.2f}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
