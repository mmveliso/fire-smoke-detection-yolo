"""Live fire and smoke detection with YOLOv4 on a Raspberry Pi.

Captures frames from the Pi camera (Picamera2), runs the trained YOLOv4 model
on the CPU through OpenCV's DNN module (Darknet itself is not needed), shows
the annotated video and logs every frame to a CSV file: detections, confidence
scores, latency and frames per second.

Expected layout on the Pi:
    ~/fire_smoke_detection/models/yolov4/yolov4-fire-smoke.cfg
    ~/fire_smoke_detection/models/yolov4/yolov4-fire-smoke_best.weights
    ~/fire_smoke_detection/models/yolov4/obj.names
    ~/fire_smoke_detection/results/                          CSV logs are written here

Usage:
    python pi_yolov4_camera_logging.py --test-name normal_room --duration 60

With the default --duration 0 the test runs until Q is pressed in the video window.
A desktop session is needed, because the script opens a video window.
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from picamera2 import Picamera2


PROJECT_DIR = Path.home() / "fire_smoke_detection"
MODEL_DIR = PROJECT_DIR / "models" / "yolov4"
RESULTS_DIR = PROJECT_DIR / "results"

CONFIG_PATH = MODEL_DIR / "yolov4-fire-smoke.cfg"
WEIGHTS_PATH = MODEL_DIR / "yolov4-fire-smoke_best.weights"
NAMES_PATH = MODEL_DIR / "obj.names"

# Same input size as training. Detections below the confidence threshold
# are discarded; NMS removes boxes that overlap a stronger one by more
# than the NMS threshold (IoU).
INPUT_WIDTH = 416
INPUT_HEIGHT = 416
CONFIDENCE_THRESHOLD = 0.31
NMS_THRESHOLD = 0.45

CLASS_COLOURS = {
    0: (0, 165, 255),  # Smoke: orange
    1: (0, 0, 255),  # Fire: red
}


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="YOLOv4 Raspberry Pi fire/smoke detection with CSV logging"
    )
    parser.add_argument(
        "--test-name",
        default="live_test",
        help="Short test label, for example normal_room or fire_video",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Test duration in seconds; use 0 to run until Q is pressed",
    )
    return parser.parse_args()


def safe_test_name(name):
    cleaned = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in name.strip()
    )
    return cleaned or "live_test"


def check_required_files():
    required_files = [CONFIG_PATH, WEIGHTS_PATH, NAMES_PATH]
    missing_files = [path for path in required_files if not path.is_file()]
    if missing_files:
        raise FileNotFoundError(
            "Missing required files: "
            + ", ".join(str(path) for path in missing_files)
        )


def load_class_names():
    with NAMES_PATH.open("r", encoding="utf-8") as names_file:
        names = [line.strip() for line in names_file if line.strip()]

    if names != ["smoke", "fire"]:
        print(
            "WARNING: Expected class order ['smoke', 'fire'], "
            f"but found {names}"
        )
    return names


def main():
    args = parse_arguments()
    test_name = safe_test_name(args.test_name)

    check_required_files()
    class_names = load_class_names()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"yolov4_{test_name}_{run_stamp}.csv"

    print("Loading YOLOv4 configuration and weights...")
    network = cv2.dnn.readNetFromDarknet(
        str(CONFIG_PATH),
        str(WEIGHTS_PATH),
    )
    # Run on the CPU with OpenCV's own DNN backend (the Pi has no CUDA GPU)
    network.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    network.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    output_layer_names = network.getUnconnectedOutLayersNames()

    # The first forward pass is slow because OpenCV initialises the network.
    # Run it once on a blank image before any timing starts.
    print("Warming up YOLOv4...")
    warmup_image = np.zeros((INPUT_HEIGHT, INPUT_WIDTH, 3), dtype=np.uint8)
    warmup_blob = cv2.dnn.blobFromImage(
        warmup_image,
        scalefactor=1 / 255.0,
        size=(INPUT_WIDTH, INPUT_HEIGHT),
        swapRB=True,
        crop=False,
    )
    network.setInput(warmup_blob)
    network.forward(output_layer_names)

    camera = Picamera2()
    # Picamera2's "RGB888" format delivers pixels in B, G, R order,
    # which is the order OpenCV expects.
    camera_configuration = camera.create_preview_configuration(
        main={"format": "RGB888", "size": (1280, 720)}
    )
    camera.configure(camera_configuration)
    camera.start()

    print("YOLOv4 loaded successfully.")
    print(f"Test: {test_name}")
    print(f"CSV log: {csv_path}")
    print("Press Q in the video window to stop early.")

    average_fps = 0.0
    frame_number = 0
    frames_with_detections = 0
    fire_frames = 0
    smoke_frames = 0
    latency_total = 0.0
    test_start = time.perf_counter()

    try:
        with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(
                [
                    "test_name",
                    "timestamp",
                    "elapsed_seconds",
                    "frame_number",
                    "detection_count",
                    "detected_classes",
                    "confidence_scores",
                    "maximum_confidence",
                    "latency_ms",
                    "fps",
                ]
            )

            while True:
                elapsed_seconds = time.perf_counter() - test_start
                if args.duration > 0 and elapsed_seconds >= args.duration:
                    break

                frame = camera.capture_array()
                # Timing starts after the frame is captured. Latency therefore covers
                # preprocessing, inference, NMS and drawing, but not camera capture
                # or the display window.
                processing_start = time.perf_counter()
                frame_height, frame_width = frame.shape[:2]

                # Scale pixels to 0-1, resize to the network input size (stretched,
                # as Darknet does by default) and swap BGR to RGB
                blob = cv2.dnn.blobFromImage(
                    frame,
                    scalefactor=1 / 255.0,
                    size=(INPUT_WIDTH, INPUT_HEIGHT),
                    swapRB=True,
                    crop=False,
                )
                network.setInput(blob)
                layer_outputs = network.forward(output_layer_names)

                boxes = []
                confidences = []
                class_ids = []

                # One output per detection scale. Each row is
                # [centre_x, centre_y, width, height, objectness, class scores...]
                # with the box values relative to the frame size.
                for output in layer_outputs:
                    for detection in output:
                        objectness = float(detection[4])
                        class_scores = detection[5:]
                        class_id = int(np.argmax(class_scores))
                        class_probability = float(class_scores[class_id])
                        confidence = objectness * class_probability

                        if confidence < CONFIDENCE_THRESHOLD:
                            continue

                        centre_x = int(detection[0] * frame_width)
                        centre_y = int(detection[1] * frame_height)
                        box_width = int(detection[2] * frame_width)
                        box_height = int(detection[3] * frame_height)
                        left = int(centre_x - box_width / 2)
                        top = int(centre_y - box_height / 2)

                        boxes.append([left, top, box_width, box_height])
                        confidences.append(confidence)
                        class_ids.append(class_id)

                # Remove overlapping boxes, keeping the most confident one
                selected_indices = cv2.dnn.NMSBoxes(
                    boxes,
                    confidences,
                    CONFIDENCE_THRESHOLD,
                    NMS_THRESHOLD,
                )

                detected_classes = []
                selected_confidences = []

                if len(selected_indices) > 0:
                    for selected_index in np.array(selected_indices).flatten():
                        left, top, box_width, box_height = boxes[selected_index]
                        class_id = class_ids[selected_index]
                        confidence = confidences[selected_index]
                        class_name = (
                            class_names[class_id]
                            if class_id < len(class_names)
                            else f"class_{class_id}"
                        )

                        detected_classes.append(class_name)
                        selected_confidences.append(confidence)

                        colour = CLASS_COLOURS.get(class_id, (0, 255, 0))
                        label = f"{class_name}: {confidence:.2f}"

                        left = max(0, left)
                        top = max(0, top)
                        right = min(frame_width - 1, left + box_width)
                        bottom = min(frame_height - 1, top + box_height)

                        cv2.rectangle(
                            frame,
                            (left, top),
                            (right, bottom),
                            colour,
                            2,
                        )
                        label_size, _ = cv2.getTextSize(
                            label,
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            2,
                        )
                        label_width, label_height = label_size
                        label_top = max(top, label_height + 10)
                        cv2.rectangle(
                            frame,
                            (left, label_top - label_height - 10),
                            (left + label_width + 10, label_top),
                            colour,
                            -1,
                        )
                        cv2.putText(
                            frame,
                            label,
                            (left + 5, label_top - 5),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            2,
                        )

                latency_ms = (time.perf_counter() - processing_start) * 1000.0
                current_fps = 1000.0 / latency_ms if latency_ms > 0 else 0.0
                # Exponential moving average, so the on-screen FPS is steady
                average_fps = (
                    current_fps
                    if average_fps == 0.0
                    else 0.9 * average_fps + 0.1 * current_fps
                )

                frame_number += 1
                latency_total += latency_ms
                unique_classes = set(detected_classes)
                if detected_classes:
                    frames_with_detections += 1
                if "fire" in unique_classes:
                    fire_frames += 1
                if "smoke" in unique_classes:
                    smoke_frames += 1

                writer.writerow(
                    [
                        test_name,
                        datetime.now().isoformat(timespec="milliseconds"),
                        f"{elapsed_seconds:.3f}",
                        frame_number,
                        len(detected_classes),
                        ";".join(detected_classes),
                        ";".join(
                            f"{value:.4f}" for value in selected_confidences
                        ),
                        f"{max(selected_confidences):.4f}"
                        if selected_confidences
                        else "",
                        f"{latency_ms:.3f}",
                        f"{current_fps:.3f}",
                    ]
                )
                # Write each row to disk immediately so the log survives an interrupted test
                csv_file.flush()

                status = (
                    f"YOLOv4 | Latency: {latency_ms:.1f} ms  "
                    f"FPS: {average_fps:.2f}  Frame: {frame_number}"
                )
                cv2.putText(
                    frame,
                    status,
                    (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("YOLOv4 Fire and Smoke Detection", frame)

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.stop()
        camera.close()
        cv2.destroyAllWindows()

    average_latency = latency_total / frame_number if frame_number else 0.0
    measured_duration = time.perf_counter() - test_start
    overall_fps = frame_number / measured_duration if measured_duration else 0.0

    print("\nTest completed")
    print(f"Frames processed: {frame_number}")
    print(f"Frames with detections: {frames_with_detections}")
    print(f"Frames containing fire: {fire_frames}")
    print(f"Frames containing smoke: {smoke_frames}")
    print(f"Average processing latency: {average_latency:.1f} ms")
    print(f"Overall processing rate: {overall_fps:.2f} FPS")
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
