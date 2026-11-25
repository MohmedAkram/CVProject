import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from shared import config
from shared.matcher import extract_features, match_images

OUT_DIR = config.ROOT / "results" / "karim_elhenawy"
SCORES_CSV = OUT_DIR / "scores.csv"
HEADER = ["query", "query_landmark", "query_split", "reference",
          "reference_landmark", "num_ratio", "num_good"]

features = {}
refs = []


def init_worker(f, r):
    global features, refs
    features, refs = f, r


def match_query(query):
    name, lm, split = query
    rows = []
    for ref_name, ref_lm in refs:
        if ref_name == name:
            continue
        res = match_images(features[name], features[ref_name])
        rows.append([name, lm, split, ref_name, ref_lm, res.num_ratio, res.num_good])
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--max-seconds", type=float)
    args = parser.parse_args()

    items = config.list_processed()
    queries = [(p.name, lm, config.split_of(ph)) for p, lm, ph in items]
    # gallery = training photos only
    gallery = [(p.name, lm) for p, lm, ph in items if config.split_of(ph) == "train"]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.fresh and SCORES_CSV.exists():
        SCORES_CSV.unlink()

    # resume from where the last run stopped
    done = set()
    if SCORES_CSV.exists():
        with open(SCORES_CSV) as f:
            done = {r["query"] for r in csv.DictReader(f)}
    todo = [q for q in queries if q[0] not in done]
    if not todo:
        print("all queries already scored")
        return
    print(f"{len(queries)} queries x {len(gallery)} references, {len(todo)} left")

    feats = {p.name: extract_features(p) for p, _, _ in items}

    new_file = not SCORES_CSV.exists()
    start = time.time()
    with open(SCORES_CSV, "a", newline="") as f, \
            Pool(args.workers, init_worker, (feats, gallery)) as pool:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(HEADER)
        for i, rows in enumerate(pool.imap(match_query, todo), 1):
            writer.writerows(rows)
            f.flush()
            if i % 10 == 0:
                print(f"{len(done) + i}/{len(queries)} done ({time.time() - start:.0f}s)")
            if args.max_seconds and time.time() - start > args.max_seconds and i < len(todo):
                print("time limit reached, run again to continue")
                pool.terminate()
                return
    print("saved", SCORES_CSV)


if __name__ == "__main__":
    main()
