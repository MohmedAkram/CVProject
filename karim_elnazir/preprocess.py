import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
from shared import config


def resize(img):
    h, w = img.shape[:2]
    scale = config.MAX_SIDE / max(h, w)
    if scale >= 1:
        return img
    return cv2.resize(img, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)


def main():
    rows = []
    for path in sorted(config.ORIGINAL_DIR.glob("*.jpeg")):
        landmark, photo = config.parse_name(path.name)
        if landmark not in config.LANDMARKS:
            continue
        img = cv2.imread(str(path))
        if img is None:
            print("could not read", path.name)
            continue
        img = resize(img)
        out = config.processed_path(landmark, photo)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        h, w = img.shape[:2]
        rows.append([path.name, landmark, photo, config.split_of(photo), w, h])

    rows.sort(key=lambda r: (r[1], r[2]))
    with open(config.MANIFEST, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "landmark", "photo", "split", "width", "height"])
        writer.writerows(rows)

    train = sum(1 for r in rows if r[3] == "train")
    print(f"processed {len(rows)} images ({train} train, {len(rows) - train} test)")


if __name__ == "__main__":
    main()
