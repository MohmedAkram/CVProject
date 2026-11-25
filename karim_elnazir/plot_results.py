# Plots good matches against viewing angle and lighting.
# It needs results/karim_elnazir/conditions.csv with the columns:
#   image_a, image_b, angle_deg, lighting
# image_a and image_b are processed file names like 1_6.jpeg.
# Only make this file if the angle and lighting were really recorded.
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from shared import config
from karim_elnazir.matcher import match_images


def find(name):
    landmark, photo = config.parse_name(name)
    return config.processed_path(landmark, photo)


def main():
    conditions = config.RESULTS_DIR / "conditions.csv"
    if not conditions.exists():
        print("no conditions.csv found, so no angle or lighting plots were made")
        return

    angles, angle_counts = [], []
    lights, light_counts = [], []
    with open(conditions) as f:
        for row in csv.DictReader(f):
            count = match_images(find(row["image_a"]), find(row["image_b"])).num_good
            if row.get("angle_deg"):
                angles.append(float(row["angle_deg"]))
                angle_counts.append(count)
            if row.get("lighting"):
                lights.append(row["lighting"])
                light_counts.append(count)

    plot_dir = config.RESULTS_DIR / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    if angles:
        plt.figure()
        plt.scatter(angles, angle_counts)
        plt.xlabel("viewing angle (degrees)")
        plt.ylabel("good matches")
        plt.savefig(plot_dir / "matches_vs_angle.png", dpi=150)
    if lights:
        plt.figure()
        plt.scatter(lights, light_counts)
        plt.xlabel("lighting")
        plt.ylabel("good matches")
        plt.savefig(plot_dir / "matches_vs_lighting.png", dpi=150)
    print("plots saved to", plot_dir)


if __name__ == "__main__":
    main()
