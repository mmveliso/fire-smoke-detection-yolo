"""Create the train / validation / test split used in this project.

The original D-Fire dataset has two parts: train (17,221 labelled images) and
test (4,306 images). The first YOLOv5 run in this project used the test set
for validation as well, so validation results were not independent of the
test set. This script fixes that:

  * the D-Fire training part is split into train (13,777) and val (3,444)
  * the D-Fire test part is kept whole as the held-out test set (4,306)

Method: sort the training image names, shuffle them with random.Random(42),
take the first 3,444 as validation and the remaining 13,777 as training.
This reproduces the split used for all reported results exactly; it was
verified against the project's original image lists. If configs/splits/val.txt
is available, the script checks the result against it before copying anything.

Expected raw layout (as downloaded from the D-Fire repository):
    DFireDataset/train/images, DFireDataset/train/labels
    DFireDataset/test/images,  DFireDataset/test/labels

Output layout:
    DFireDataset_split/images/{train,val,test}
    DFireDataset_split/labels/{train,val,test}

Run scripts/audit_and_correct_labels.py afterwards.
"""

from pathlib import Path
import random
import shutil

ROOT = Path("/content/drive/MyDrive/Fire_smoke_detection")
SOURCE = ROOT / "data" / "raw" / "DFireDataset"
DEST = ROOT / "data" / "processed" / "DFireDataset_split"

# Optional: the published validation list, used to verify the split.
VAL_LIST = Path(__file__).resolve().parent.parent / "configs" / "splits" / "val.txt"

SEED = 42
VAL_COUNT = 3444
EXPECTED_TRAIN_PART = 17221
EXPECTED_TEST_PART = 4306

EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def labelled_images(part):
    """Sorted images of one raw part that have a label file."""
    image_dir = SOURCE / part / "images"
    label_dir = SOURCE / part / "labels"

    images = sorted(
        (p for p in image_dir.iterdir()
         if p.is_file() and p.suffix.lower() in EXTENSIONS),
        key=lambda p: p.name,
    )

    return [p for p in images if (label_dir / f"{p.stem}.txt").exists()]


def copy_split(images, split, part):
    """Copy images and their labels into the split folders."""
    image_dest = DEST / "images" / split
    label_dest = DEST / "labels" / split

    image_dest.mkdir(parents=True, exist_ok=True)
    label_dest.mkdir(parents=True, exist_ok=True)

    for image in images:
        label = SOURCE / part / "labels" / f"{image.stem}.txt"
        shutil.copy2(image, image_dest / image.name)
        shutil.copy2(label, label_dest / label.name)


def main():
    train_part = labelled_images("train")
    test_part = labelled_images("test")

    print("Labelled images in raw train part:", len(train_part))
    print("Labelled images in raw test part:", len(test_part))

    if len(train_part) != EXPECTED_TRAIN_PART or len(test_part) != EXPECTED_TEST_PART:
        raise RuntimeError(
            "Unexpected image count. The split is reproducible only from the "
            f"standard D-Fire release ({EXPECTED_TRAIN_PART} train, "
            f"{EXPECTED_TEST_PART} test)."
        )

    # Shuffle the sorted training images with a fixed seed.
    shuffled = list(train_part)
    random.Random(SEED).shuffle(shuffled)

    validation = shuffled[:VAL_COUNT]
    training = shuffled[VAL_COUNT:]

    # Verify against the published validation list before copying anything.
    if VAL_LIST.exists():
        published = {
            line.strip() for line in VAL_LIST.read_text().splitlines()
            if line.strip()
        }
        if {p.name for p in validation} != published:
            raise RuntimeError(
                "The generated validation set differs from configs/splits/val.txt."
            )
        print("Validation set matches configs/splits/val.txt")

    if DEST.exists() and any(DEST.iterdir()):
        raise RuntimeError(f"{DEST} already exists and is not empty.")

    copy_split(training, "train", "train")
    copy_split(validation, "val", "train")
    copy_split(test_part, "test", "test")

    print("Train:", len(training))
    print("Validation:", len(validation))
    print("Test:", len(test_part))


if __name__ == "__main__":
    main()
