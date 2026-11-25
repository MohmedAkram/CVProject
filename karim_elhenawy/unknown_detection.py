import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared import config
from karim_elhenawy.dataset_split import load_splits
from karim_elhenawy.knn_classifier import load_scores, only, PARAMS_JSON

OUT_DIR = config.ROOT / "results" / "karim_elhenawy"
UNKNOWN_CSV = OUT_DIR / "unknown_detection.csv"
UNKNOWN_TXT = OUT_DIR / "unknown_metrics.txt"
UNKNOWN = "unknown place"


def best_score(candidates, hide=None):
    # hide = landmark removed from the gallery to fake an unknown place
    return max((c[2] for c in candidates if c[1] != hide), default=0)


def rates(known, unknown, t):
    accepted = sum(s >= t for s in known) / len(known)
    rejected = sum(s < t for s in unknown) / len(unknown)
    return accepted, rejected, (accepted + rejected) / 2


def pick_threshold(known, unknown):
    options = sorted(set(known + unknown + [0]))
    options.append(options[-1] + 1)
    # best balanced accuracy, smallest threshold on ties
    return max(options, key=lambda t: (rates(known, unknown, t)[2], -t))


def main():
    splits = load_splits()
    scores = load_scores()
    train = [r for r in splits if r["split"] == "train"]
    test = [r for r in splits if r["split"] == "test"]
    train_names = {r["filename"] for r in train}

    # threshold from the training folds only
    known, unknown = [], []
    for r in train:
        gallery = {x["filename"] for x in train if x["fold"] != r["fold"]}
        cands = only(scores[r["filename"]], gallery)
        known.append(best_score(cands))
        unknown.append(best_score(cands, hide=r["landmark"]))
    t = pick_threshold(known, unknown)
    train_rates = rates(known, unknown, t)

    # test with the fixed threshold
    rows, test_known, test_unknown = [], [], []
    for r in test:
        cands = only(scores[r["filename"]], train_names)
        k_score = best_score(cands)
        u_score = best_score(cands, hide=r["landmark"])
        test_known.append(k_score)
        test_unknown.append(u_score)
        rows.append([r["filename"], r["landmark"],
                     k_score, "known" if k_score >= t else UNKNOWN,
                     u_score, "known" if u_score >= t else UNKNOWN])
    test_rates = rates(test_known, test_unknown, t)

    with open(UNKNOWN_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["test_image", "true_landmark",
                         "best_score_full_gallery", "decision_full_gallery (should be known)",
                         "best_score_own_landmark_hidden", "decision_own_landmark_hidden (should be unknown place)"])
        writer.writerows(rows)

    n_acc = sum(s >= t for s in test_known)
    n_rej = sum(s < t for s in test_unknown)
    lines = [
        "Unknown-place detection (optional bonus)",
        "Confidence: num_good of the best-matching training photo",
        f"Threshold: score < {t} -> '{UNKNOWN}' (chosen on training folds only)",
        "",
        "Training folds (used to choose the threshold):",
        f"  known photos accepted:   {train_rates[0]:.3f} ({len(known)} photos)",
        f"  unknown photos rejected: {train_rates[1]:.3f} ({len(unknown)} simulated)",
        f"  balanced accuracy:       {train_rates[2]:.3f}",
        "",
        "Test photos (threshold fixed beforehand):",
        f"  known photos accepted:   {test_rates[0]:.3f} ({n_acc}/{len(test)})",
        f"  unknown photos rejected: {test_rates[1]:.3f} ({n_rej}/{len(test)})",
        f"  balanced accuracy:       {test_rates[2]:.3f}",
        "",
        "Unknown places are simulated by removing the photo's own landmark from the gallery.",
    ]
    UNKNOWN_TXT.write_text("\n".join(lines) + "\n")

    if PARAMS_JSON.exists():
        params = json.loads(PARAMS_JSON.read_text())
        params["unknown_threshold"] = t
        PARAMS_JSON.write_text(json.dumps(params, indent=2))

    print("\n".join(lines))


if __name__ == "__main__":
    main()
