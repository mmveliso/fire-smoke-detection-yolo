"""Prepare the split dataset for Darknet (YOLOv4).

Darknet needs two things that YOLOv5 does not:

  1. A text file per split listing the absolute path of every image.
  2. Each label file in the same folder as its image
     (images/train/x.jpg and images/train/x.txt).

This script writes configs/YOLOv4/train.txt, valid.txt and test.txt and
copies every label next to its image. Run it after
scripts/audit_and_correct_labels.py so the corrected labels are the ones copied.

It was written for publication to reproduce those files; the lists it writes
have the same form as the ones used in the project (sorted absolute paths).
"""

from pathlib import Path
import shutil

ROOT = Path("/content/drive/MyDrive/Fire_smoke_detection")
DATASET = ROOT / "data" / "processed" / "DFireDataset_split"
CONFIG_DIR = ROOT / "configs" / "YOLOv4"

# split folder name -> Darknet list file name
LIST_NAMES = {"train": "train.txt", "val": "valid.txt", "test": "test.txt"}
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def main():
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    for split, list_name in LIST_NAMES.items():
        image_dir = DATASET / "images" / split
        label_dir = DATASET / "labels" / split

        images = sorted(
            (p for p in image_dir.iterdir()
             if p.is_file() and p.suffix.lower() in EXTENSIONS),
            key=lambda p: p.name,
        )

        for image in images:
            label = label_dir / f"{image.stem}.txt"

            if not label.exists():
                raise FileNotFoundError(label)

            # Darknet looks for the label beside the image.
            shutil.copy2(label, image_dir / label.name)

        list_file = CONFIG_DIR / list_name
        list_file.write_text("\n".join(str(p) for p in images) + "\n")

        print(f"{split}: {len(images)} images -> {list_file}")


if __name__ == "__main__":
    main()
