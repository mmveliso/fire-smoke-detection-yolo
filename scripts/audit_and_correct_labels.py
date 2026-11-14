"""Audit the YOLO label files of the split dataset and correct invalid ones.

A valid annotation line is:  class_id x_center y_center width height
with class_id 0 (smoke) or 1 (fire) and all four box values normalised to 0-1.

Two kinds of invalid annotation exist in D-Fire:

  * zero-area boxes (width or height is 0)        -> the line is removed
  * boxes wider or taller than the image (> 1)    -> the box is clipped to
                                                     the image borders

On the standard D-Fire release this changes 26 label files: 18 annotations
removed and 8 clipped (13 files in train, 1 in val, 12 in test). The exact
changes made for this project are recorded in results/correction_log.txt.

This script was written for publication from that log. Its clipping was
checked against all 8 clipped annotations in the log and gives the same
corrected values.

By default the script only reports. Set APPLY = True to write the corrections;
every changed file is first backed up to BACKUP_DIR.
"""

from pathlib import Path
from datetime import datetime
import shutil

ROOT = Path("/content/drive/MyDrive/Fire_smoke_detection")
DATASET = ROOT / "data" / "processed" / "DFireDataset_split"
BACKUP_DIR = ROOT / "data" / "processed" / "DFire_invalid_labels_before_correction"
LOG_FILE = BACKUP_DIR / "correction_log.txt"

SPLITS = ["train", "val", "test"]
VALID_CLASSES = {0, 1}
APPLY = False


def check_line(line):
    """Return (action, corrected_line, reason) for one annotation line.

    action is "keep", "remove", "clip" or "error" (cannot be fixed automatically).
    """
    fields = line.split()

    if len(fields) != 5:
        return "error", None, "expected 5 fields"

    try:
        class_value, x, y, w, h = (float(value) for value in fields)
    except ValueError:
        return "error", None, "non-numeric value"

    if not class_value.is_integer() or int(class_value) not in VALID_CLASSES:
        return "error", None, "invalid class ID"

    # A box with no width or no height contains nothing.
    if w <= 0 or h <= 0:
        return "remove", None, "zero-area box"

    out_of_range = (
        w > 1 or h > 1
        or not 0 <= x <= 1
        or not 0 <= y <= 1
    )

    if not out_of_range:
        return "keep", line, ""

    # Convert to corners, clip the corners to the image, convert back.
    x_min = max(0.0, x - w / 2)
    y_min = max(0.0, y - h / 2)
    x_max = min(1.0, x + w / 2)
    y_max = min(1.0, y + h / 2)

    new_w = x_max - x_min
    new_h = y_max - y_min

    if new_w <= 0 or new_h <= 0:
        return "remove", None, "box lies outside the image"

    corrected = (
        f"{int(class_value)} "
        f"{(x_min + x_max) / 2:.10f} {(y_min + y_max) / 2:.10f} "
        f"{new_w:.10f} {new_h:.10f}"
    )

    return "clip", corrected, "box exceeds the image"


def main():
    log_lines = []
    counts = {"remove": 0, "clip": 0, "error": 0}
    affected_files = 0

    for split in SPLITS:
        label_dir = DATASET / "labels" / split

        for label_path in sorted(label_dir.glob("*.txt")):
            original = label_path.read_text().splitlines()
            corrected = []
            file_log = []

            for line_number, raw_line in enumerate(original, start=1):
                line = raw_line.strip()

                if not line:
                    continue

                action, new_line, reason = check_line(line)

                if action == "keep":
                    corrected.append(line)
                elif action == "remove":
                    counts["remove"] += 1
                    file_log.append(f"Line {line_number} REMOVED: {line}")
                elif action == "clip":
                    counts["clip"] += 1
                    corrected.append(new_line)
                    file_log.append(
                        f"Line {line_number} CLIPPED:\n"
                        f" Before: {line}\n"
                        f" After: {new_line}"
                    )
                else:
                    # Left unchanged: needs a manual decision.
                    counts["error"] += 1
                    corrected.append(line)
                    file_log.append(f"Line {line_number} NOT FIXED ({reason}): {line}")

            if not file_log:
                continue

            affected_files += 1
            log_lines.append(f"===== {split}/{label_path.name} =====")
            log_lines.extend(file_log)

            if APPLY:
                backup_path = BACKUP_DIR / "labels" / split / label_path.name
                backup_path.parent.mkdir(parents=True, exist_ok=True)

                if not backup_path.exists():
                    shutil.copy2(label_path, backup_path)

                label_path.write_text(
                    "\n".join(corrected) + ("\n" if corrected else "")
                )

    header = [
        "D-Fire annotation correction log",
        f"Date: {datetime.now().isoformat()}",
        f"Dataset: {DATASET}",
        f"Affected label files: {affected_files}",
        f"Zero-area annotations removed: {counts['remove']}",
        f"Out-of-range annotations clipped: {counts['clip']}",
        f"Annotations needing manual review: {counts['error']}",
        f"Corrections written: {APPLY}",
        "",
    ]

    report = "\n".join(header + log_lines) + "\n"
    print(report)

    if APPLY:
        BACKUP_DIR.mkdir(parents=True, exist_ok=True)
        LOG_FILE.write_text(report)
        print("Log saved to:", LOG_FILE)


if __name__ == "__main__":
    main()
