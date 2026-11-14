"""Live fire and smoke detection with YOLOv5s on a Raspberry Pi.

Captures frames from the Pi camera (Picamera2), runs the trained YOLOv5s model
on the CPU, shows the annotated video and logs every frame to a CSV file:
detections, confidence scores, latency and frames per second.

Expected layout on the Pi:
    ~/fire_smoke_detection/yolov5/                           clone of ultralytics/yolov5
    ~/fire_smoke_detection/models/yolov5s_fire_smoke_best.pt trained weights (best.pt)
    ~/fire_smoke_detection/results/                          CSV logs are written here

Usage:
    python pi_yolov5s_camera_logging.py --test-name normal_room --duration 60

Use --duration 0 to run until Q is pressed in the video window.
A desktop session is needed, because the script opens a video window.
"""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import cv2
import torch
from picamera2 import Picamera2

PROJECT_DIR = Path.home() / "fire_smoke_detection"
YOLO_DIR = PROJECT_DIR / "yolov5"
WEIGHTS = PROJECT_DIR / "models" / "yolov5s_fire_smoke_best.pt"
RESULTS_DIR = PROJECT_DIR / "results"

# The YOLOv5 repository is not an installed package, so add it to the
# import path before importing its modules.
sys.path.insert(0, str(YOLO_DIR))

from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.plots import Annotator, colors
from utils.torch_utils import select_device


# Same input size as training. Detections below the confidence threshold
# are discarded; NMS removes boxes that overlap a stronger one by more
# than the NMS threshold (IoU).
IMAGE_SIZE = 416
CONFIDENCE_THRESHOLD = 0.31
NMS_THRESHOLD = 0.45


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="YOLOv5 live fire/smoke detection with CSV logging"
    )
    parser.add_argument(
        "--test-name",
        default="live_test",
        help="Short test label, for example normal_room or smoke_video",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Test duration in seconds; use 0 to run until Q is pressed",
    )
    return parser.parse_args()


def safe_test_name(name):
    cleaned = "".join(
        character if character.isalnum() or character in "-_" else "_"
        for character in name.strip()
    )
    return cleaned or "live_test"


def main():
    args = parse_arguments()
    test_name = safe_test_name(args.test_name)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = RESULTS_DIR / f"{test_name}_{run_stamp}.csv"

    # The Raspberry Pi has no CUDA GPU, so run on the CPU in full precision
    device = select_device("cpu")
    model = DetectMultiBackend(
        WEIGHTS,
        device=device,
        dnn=False,
        fp16=False,
    )

    stride = model.stride
    class_names = model.names
    using_pytorch = model.pt
    # Make sure the input size is a multiple of the model stride, then run
    # one dummy inference so the first real frame is not slowed by start-up
    image_size = check_img_size((IMAGE_SIZE, IMAGE_SIZE), s=stride)
    model.warmup(imgsz=(1, 3, *image_size))

    camera = Picamera2()
    # Picamera2's "RGB888" format delivers pixels in B, G, R order,
    # which is the order OpenCV expects.
    camera_configuration = camera.create_preview_configuration(
        main={"format": "RGB888", "size": (1280, 720)}
    )
    camera.configure(camera_configuration)
    camera.start()

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
                inference_start = time.perf_counter()

                # Resize to the model input size with padding, keeping the aspect ratio
                input_image = letterbox(
                    frame,
                    image_size,
                    stride=stride,
                    auto=using_pytorch,
                )[0]
                # Height-width-channel to channel-height-width, and BGR to RGB
                input_image = input_image.transpose((2, 0, 1))[::-1]
                input_image = input_image.copy()
                input_tensor = torch.from_numpy(input_image).to(device)
                # Scale pixel values from 0-255 to 0-1
                input_tensor = input_tensor.float() / 255.0

                if input_tensor.ndim == 3:
                    input_tensor = input_tensor.unsqueeze(0)

                predictions = model(input_tensor)
                predictions = non_max_suppression(
                    predictions,
                    CONFIDENCE_THRESHOLD,
                    NMS_THRESHOLD,
                )

                annotator = Annotator(
                    frame,
                    line_width=2,
                    example=str(class_names),
                )
                detected_classes = []
                confidence_scores = []

                for detections in predictions:
                    if len(detections):
                        # Map the boxes from the resized input back to the original frame
                        detections[:, :4] = scale_boxes(
                            input_tensor.shape[2:],
                            detections[:, :4],
                            frame.shape,
                        ).round()

                        for *box, confidence, class_id in reversed(detections):
                            class_number = int(class_id)
                            class_name = str(class_names[class_number])
                            confidence_value = float(confidence)
                            detected_classes.append(class_name)
                            confidence_scores.append(confidence_value)

                            annotator.box_label(
                                box,
                                f"{class_name} {confidence_value:.2f}",
                                color=colors(class_number, True),
                            )

                latency_ms = (time.perf_counter() - inference_start) * 1000.0
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
                        ";".join(f"{value:.4f}" for value in confidence_scores),
                        f"{max(confidence_scores):.4f}"
                        if confidence_scores
                        else "",
                        f"{latency_ms:.3f}",
                        f"{current_fps:.3f}",
                    ]
                )
                # Write each row to disk immediately so the log survives an interrupted test
                csv_file.flush()

                displayed_frame = annotator.result()
                status = (
                    f"Latency: {latency_ms:.1f} ms  "
                    f"FPS: {average_fps:.2f}  "
                    f"Frame: {frame_number}"
                )
                cv2.putText(
                    displayed_frame,
                    status,
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow(
                    "YOLOv5 Fire and Smoke Detection",
                    displayed_frame,
                )

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
    print(f"Average inference latency: {average_latency:.1f} ms")
    print(f"Overall processing rate: {overall_fps:.2f} FPS")
    print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
