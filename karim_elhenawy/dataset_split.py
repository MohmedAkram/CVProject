import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared import config

N_FOLDS = 5
SEED = 42
OUT_DIR = config.ROOT / "results" / "karim_elhenawy"
SPLITS_CSV = OUT_DIR / "splits.csv"


def make_folds(n_folds=N_FOLDS, seed=SEED):
    rng = random.Random(seed)
    rows = []
    start = 0  # rotate so leftover photos don't all go to fold 0

    for lm in config.LANDMARKS:
        photos = [(p, ph) for p, l, ph in config.list_processed() if l == lm]
        train = [x for x in photos if config.split_of(x[1]) == "train"]
        test = [x for x in photos if config.split_of(x[1]) == "test"]

        rng.shuffle(train)
        for i, (p, ph) in enumerate(train):
            rows.append({"filename": p.name, "landmark": lm, "photo": ph,
                         "split": "train", "fold": (start + i) % n_folds})
        start = (start + len(train)) % n_folds

        for p, ph in test:
            rows.append({"filename": p.name, "landmark": lm, "photo": ph,
                         "split": "test", "fold": -1})

    rows.sort(key=lambda r: (r["landmark"], r["photo"]))
    return rows


def load_splits():
    with open(SPLITS_CSV) as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["landmark"] = int(r["landmark"])
        r["photo"] = int(r["photo"])
        r["fold"] = int(r["fold"])
    return rows


def fold_sets(rows, fold):
    train = [r for r in rows if r["split"] == "train"]
    gallery = [r for r in train if r["fold"] != fold]
    val = [r for r in train if r["fold"] == fold]
    return gallery, val


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=N_FOLDS)
    args = parser.parse_args()

    rows = make_folds(args.folds)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SPLITS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "landmark", "photo", "split", "fold"])
        writer.writeheader()
        writer.writerows(rows)

    # photos per landmark in each fold
    print("landmark  test  " + "  ".join(f"f{k}" for k in range(args.folds)) + "  train")
    for lm in config.LANDMARKS:
        mine = [r for r in rows if r["landmark"] == lm]
        test = sum(r["split"] == "test" for r in mine)
        folds = [sum(r["fold"] == k for r in mine) for k in range(args.folds)]
        print(f"{lm:>8}  {test:>4}  " + "  ".join(f"{n:>2}" for n in folds) + f"  {sum(folds):>5}")
    print("saved", SPLITS_CSV)


if __name__ == "__main__":
    main()
