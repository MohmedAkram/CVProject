import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared import config
from karim_elhenawy.dataset_split import load_splits

OUT_DIR = config.ROOT / "results" / "karim_elhenawy"
SCORES_CSV = OUT_DIR / "scores.csv"
CV_CSV = OUT_DIR / "cv_results.csv"
PARAMS_JSON = OUT_DIR / "chosen_params.json"

K_VALUES = [1, 3, 5, 7, 9, 11, 15]
RULES = ["majority", "weighted"]
TIE_RULE = "larger total num_good among the top k, then highest-ranked neighbour"


def load_scores():
    # query -> list of (reference, reference landmark, num_good, num_ratio)
    scores = defaultdict(list)
    with open(SCORES_CSV) as f:
        for r in csv.DictReader(f):
            scores[r["query"]].append((r["reference"], int(r["reference_landmark"]),
                                       int(r["num_good"]), int(r["num_ratio"])))
    return scores


def only(candidates, allowed):
    return [c for c in candidates if c[0] in allowed]


def predict(candidates, k, rule="majority"):
    top = sorted(candidates, key=lambda c: (-c[2], -c[3], c[0]))[:k]

    votes = defaultdict(float)
    total = defaultdict(int)
    for _, lm, good, _ in top:
        votes[lm] += 1 if rule == "majority" else good
        total[lm] += good

    best = [lm for lm in votes if votes[lm] == max(votes.values())]
    # tie: more total matches, then the closest neighbour
    if len(best) > 1:
        best = [lm for lm in best if total[lm] == max(total[x] for x in best)]
    if len(best) > 1:
        best = [next(lm for _, lm, _, _ in top if lm in best)]
    return best[0], dict(votes), top


def cross_validate(splits, scores):
    train = [r for r in splits if r["split"] == "train"]
    folds = sorted({r["fold"] for r in train})
    results = []

    for rule in RULES:
        for k in K_VALUES:
            fold_acc = []
            correct = 0
            for f in folds:
                gallery = {r["filename"] for r in train if r["fold"] != f}
                val = [r for r in train if r["fold"] == f]
                hits = sum(predict(only(scores[r["filename"]], gallery), k, rule)[0] == r["landmark"]
                           for r in val)
                fold_acc.append(hits / len(val))
                correct += hits
            results.append({"rule": rule, "k": k, "acc": correct / len(train), "folds": fold_acc})
    return results


def load_params():
    with open(PARAMS_JSON) as f:
        return json.load(f)


def main():
    results = cross_validate(load_splits(), load_scores())

    with open(CV_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["rule", "k", "cv_accuracy"] + [f"fold_{i}" for i in range(len(results[0]["folds"]))])
        for r in results:
            writer.writerow([r["rule"], r["k"], round(r["acc"], 4)] + [round(a, 4) for a in r["folds"]])

    # best accuracy, then smaller k, then majority
    best = sorted(results, key=lambda r: (-r["acc"], r["k"], RULES.index(r["rule"])))[0]
    params = {"k": best["k"], "rule": best["rule"], "cv_accuracy": round(best["acc"], 4),
              "n_folds": len(best["folds"]), "tie_rule": TIE_RULE}
    with open(PARAMS_JSON, "w") as f:
        json.dump(params, f, indent=2)

    for r in results:
        print(f"{r['rule']:>9}  k={r['k']:<3} {r['acc']:.3f}" + ("  <- chosen" if r is best else ""))
    print("saved", CV_CSV, "and", PARAMS_JSON)


if __name__ == "__main__":
    main()
