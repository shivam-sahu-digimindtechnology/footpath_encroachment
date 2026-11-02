"""
Footpath Encroachment Detection System
--------------------------------------
Pipeline:
  1. Segment Footpath  → binary mask
  2. Detect objects    → stalls + debris + occupancy + YOLOv8
  3. Deduplicate
  4. BBox ∩ Footpath mask → overlap ratio → ENCROACHMENT YES/NO
  5. Visualize & print
"""

import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np

from config import settings
from detector import FootpathDetector, MultiModelDetector
from bounding_box import (
    filter_duplicates,
    calculate_mask_overlap,
    overlay_footpath_mask,
    draw_bbox,
)


def classify_detection(
    det: Dict[str, Any],
    footpath_mask: Optional[np.ndarray],
) -> Dict[str, Any]:
    """
    Decide status for one detection:
      - IGNORED          (person / dog / infrastructure)
      - ENCROACHMENT
      - NOT_ENCROACHMENT
    """
    name_lower = det["class_name"].lower().strip()

    # 1. Explicit ignore list
    if name_lower in [c.lower() for c in settings.IGNORE_CLASSES]:
        det["status"] = "IGNORED"
        det["overlap_ratio"] = 0.0
        return det

    # 2. Infrastructure list
    if name_lower in [c.lower() for c in settings.INFRASTRUCTURE_CLASSES]:
        det["status"] = "IGNORED"
        det["overlap_ratio"] = 0.0
        return det

    # 3. No footpath → cannot be encroachment
    if footpath_mask is None:
        det["status"] = "NOT_ENCROACHMENT"
        det["overlap_ratio"] = 0.0
        return det

    # 4. Compute overlap
    overlap_info = calculate_mask_overlap(det["bbox"], footpath_mask)
    det["overlap_ratio"] = overlap_info["overlap_ratio"]
    det["intersection_pixels"] = overlap_info["intersection_pixels"]

    if overlap_info["is_encroachment"]:
        det["status"] = "ENCROACHMENT"
    else:
        det["status"] = "NOT_ENCROACHMENT"

    return det


def print_detection(det: Dict[str, Any]) -> None:
    status = det["status"]
    print("-" * 40)
    print("OBJECT DETECTED")
    print("-" * 40)
    print(f"Model       : {det.get('model', '-')}")
    print(f"Class       : {det['class_name']}")
    print(f"Confidence  : {det['confidence']:.2f}")
    print(f"BBox        : {tuple(det['bbox'])}")

    if status == "IGNORED":
        print("Encroachment: IGNORED")
    else:
        print(f"Overlap     : {det.get('overlap_ratio', 0) * 100:.1f}%")
        print(f"Encroachment: {'YES' if status == 'ENCROACHMENT' else 'NO'}")
    print("-" * 40)


def process_frame(
    frame: np.ndarray,
    footpath_det: FootpathDetector,
    object_det: MultiModelDetector,
) -> np.ndarray:
    """Full pipeline for one image/frame. Returns annotated frame."""

    # ---------------------------------------------------------------
    # 1. Footpath segmentation
    # ---------------------------------------------------------------
    fp_result = footpath_det.detect(frame)
    footpath_mask = fp_result["mask"]
    fp_conf = fp_result["confidence"]

    if footpath_mask is not None:
        print(f"Footpath detected | confidence={fp_conf:.2f}")
    else:
        print("Footpath not detected")

    # ---------------------------------------------------------------
    # 2. Object detection (all models)
    # ---------------------------------------------------------------
    raw_dets = object_det.detect_all(frame)

    # ---------------------------------------------------------------
    # 3. Duplicate filtering
    # ---------------------------------------------------------------
    dets = filter_duplicates(raw_dets, settings.DUPLICATE_IOU_THRESHOLD)

    # ---------------------------------------------------------------
    # 4. Encroachment classification
    # ---------------------------------------------------------------
    classified = [classify_detection(d, footpath_mask) for d in dets]

    # ---------------------------------------------------------------
    # 5. Console output
    # ---------------------------------------------------------------
    if settings.PRINT_DETECTIONS:
        for d in classified:
            print_detection(d)

    # ---------------------------------------------------------------
    # 6. Visualization
    # ---------------------------------------------------------------
    annotated = overlay_footpath_mask(frame, footpath_mask)

    # Optional: draw footpath boxes for debug
    for box in fp_result.get("boxes", []):
        x1, y1, x2, y2 = box
        cv2.rectangle(
            annotated, (x1, y1), (x2, y2),
            settings.COLOR_FOOTPATH_BOX, 1,
        )

    for d in classified:
        draw_bbox(annotated, d)

    # Status bar
    n_enc = sum(1 for d in classified if d["status"] == "ENCROACHMENT")
    status_line = f"Encroachments: {n_enc} | Objects: {len(classified)}"
    if footpath_mask is None:
        status_line = "NO FOOTPATH | " + status_line

    cv2.putText(
        annotated,
        status_line,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return annotated


def run_image(footpath_det: FootpathDetector, object_det: MultiModelDetector) -> None:
    path = settings.INPUT_IMAGE
    print(f"\n[IMAGE] Loading: {path}")
    frame = cv2.imread(path)
    if frame is None:
        print(f"[FATAL] Cannot read image: {path}")
        sys.exit(1)

    annotated = process_frame(frame, footpath_det, object_det)

    if settings.SAVE_OUTPUT:
        out_path = Path(settings.OUTPUT_DIR) / "result_image.jpg"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), annotated)
        print(f"[OK] Saved → {out_path}")

    if settings.SHOW_WINDOW:
        cv2.imshow("Footpath Encroachment Detection", annotated)
        print("Press any key to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_video(footpath_det: FootpathDetector, object_det: MultiModelDetector) -> None:
    path = settings.INPUT_VIDEO
    print(f"\n[VIDEO] Opening: {path}")
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        print(f"[FATAL] Cannot open video: {path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = None
    if settings.SAVE_OUTPUT:
        Path(settings.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(settings.OUTPUT_PATH, fourcc, fps, (width, height))
        print(f"[OK] Writing output → {settings.OUTPUT_PATH}")

    frame_idx = 0
    t0 = time.time()

    print("\n[INFO] Processing video... Press 'q' to quit.\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            start = time.time()

            annotated = process_frame(frame, footpath_det, object_det)

            # FPS overlay
            if settings.SHOW_FPS:
                elapsed = time.time() - start
                fps_text = f"FPS: {1.0 / elapsed:.1f}" if elapsed > 0 else "FPS: --"
                cv2.putText(
                    annotated, fps_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA,
                )

            if writer is not None:
                writer.write(annotated)

            if settings.SHOW_WINDOW:
                cv2.imshow("Footpath Encroachment Detection", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    print("[INFO] Quit by user.")
                    break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        if settings.SHOW_WINDOW:
            cv2.destroyAllWindows()

        total = time.time() - t0
        print(f"\nProcessed {frame_idx} frames in {total:.1f}s "
              f"({frame_idx / total:.1f} FPS avg)")


def main():
    print("=" * 60)
    print("  FOOTPATH ENCROACHMENT DETECTION SYSTEM")
    print("=" * 60)
    print(f"Device              : {settings.DEVICE}")
    print(f"Footpath overlap th : {settings.FOOTPATH_OVERLAP_THRESHOLD}")
    print(f"Duplicate IoU th    : {settings.DUPLICATE_IOU_THRESHOLD}")
    print(f"Mode                : {'VIDEO' if settings.USE_VIDEO else 'IMAGE'}")
    print("-" * 60)

    # Load models once
    footpath_det = FootpathDetector()
    object_det = MultiModelDetector()

    if settings.USE_VIDEO:
        run_video(footpath_det, object_det)
    else:
        run_image(footpath_det, object_det)

    print("\nDone.")
    print("=" * 60)


if __name__ == "__main__":
    main()