from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ORIGINAL_DIR = ROOT / "OriginalImages"
PROCESSED_DIR = ROOT / "dataset" / "processed"
MANIFEST = ROOT / "dataset" / "manifest.csv"
RESULTS_DIR = ROOT / "results" / "karim_elnazir"

LANDMARKS = [1, 2, 3, 4, 5, 6]

# longest side of every processed image, in pixels
MAX_SIDE = 800

# photos _1 to _5 of every landmark are test, everything else is train
TEST_PHOTOS = range(1, 6)

# matcher settings
RATIO = 0.75
RANSAC_THRESH = 5.0
MIN_MATCHES = 8


def parse_name(name):
    landmark, photo = Path(name).stem.split("_")
    return int(landmark), int(photo)


def split_of(photo):
    return "test" if photo in TEST_PHOTOS else "train"


def processed_path(landmark, photo):
    return PROCESSED_DIR / f"landmark_{landmark}" / f"{landmark}_{photo}.jpeg"


def list_processed(split=None):
    items = []
    for landmark in LANDMARKS:
        folder = PROCESSED_DIR / f"landmark_{landmark}"
        for path in folder.glob("*.jpeg"):
            _, photo = parse_name(path.name)
            if split is None or split_of(photo) == split:
                items.append((path, landmark, photo))
    return sorted(items, key=lambda x: (x[1], x[2]))
