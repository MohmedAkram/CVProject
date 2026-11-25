# Turns the Harris vs SIFT results into pictures.
# Run compare_detectors.py first, then this file.
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shared import config

OUT = config.RESULTS_DIR / "plots"


def read_summary():
    with open(config.RESULTS_DIR / "detector_comparison.csv") as f:
        return {row["detector"]: row for row in csv.DictReader(f)}


def read_pairs():
    with open(config.RESULTS_DIR / "detector_comparison_pairs.csv") as f:
        return list(csv.DictReader(f))


def summary_bars():
    summary = read_summary()
    metrics = [
        ("mean_keypoints", "Keypoints per image"),
        ("mean_grid_coverage", "Image coverage (0 to 1)"),
        ("mean_repeatability", "Repeatability (0 to 1)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (key, title) in zip(axes, metrics):
        values = [float(summary["harris"][key]), float(summary["sift"][key])]
        bars = ax.bar(["Harris", "SIFT"], values, color=["tab:red", "tab:green"])
        ax.set_title(title)
        ax.bar_label(bars, fmt="%.2f" if values[0] < 10 else "%.0f")
    fig.suptitle("Harris vs SIFT (higher is better in all three)")
    fig.tight_layout()
    fig.savefig(OUT / "detector_summary.png", dpi=150)
    plt.close(fig)


def repeatability_per_pair():
    pairs = read_pairs()
    labels = [f"{p['image_a'][:-5]} / {p['image_b'][:-5]}" for p in pairs]
    x = range(len(pairs))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.bar([i - 0.2 for i in x], [float(p["harris_repeatability"]) for p in pairs], 0.4,
           label="Harris", color="tab:red")
    ax.bar([i + 0.2 for i in x], [float(p["sift_repeatability"]) for p in pairs], 0.4,
           label="SIFT", color="tab:green")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel("repeatability")
    ax.set_title("Repeatability for each pair of photos")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "repeatability_per_pair.png", dpi=150)
    plt.close(fig)


def keypoint_examples():
    folder = config.RESULTS_DIR / "keypoint_images"
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, landmark in enumerate([1, 3, 5]):
        for row, name in enumerate(["harris", "sift"]):
            img = cv2.imread(str(folder / f"{landmark}_6_{name}.jpg"))
            axes[row][col].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axes[row][col].set_title(f"Landmark {landmark}: {name.upper()}")
            axes[row][col].axis("off")
    fig.tight_layout()
    fig.savefig(OUT / "keypoint_examples.png", dpi=120)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    summary_bars()
    repeatability_per_pair()
    keypoint_examples()
    print("saved to", OUT)


if __name__ == "__main__":
    main()
