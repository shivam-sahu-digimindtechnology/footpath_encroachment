

from typing import List, Dict, Any, Tuple, Optional
import cv2
import numpy as np
from config import settings


# ------------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------------

def calculate_bbox_area(bbox: List[int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def calculate_iou(box_a: List[int], box_b: List[int]) -> float:
    """Intersection-over-Union of two [x1,y1,x2,y2] boxes."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    if inter_area == 0:
        return 0.0

    area_a = calculate_bbox_area(box_a)
    area_b = calculate_bbox_area(box_b)
    union = area_a + area_b - inter_area
    return inter_area / union if union > 0 else 0.0


def merge_footpath_masks(
    masks: List[np.ndarray],
    frame_shape: Tuple[int, int],
) -> np.ndarray:
    """
    Combine multiple Footpath instance masks into one binary mask.
    frame_shape = (height, width)
    Returns uint8 mask: 0 = outside, 255 = footpath.
    """
    h, w = frame_shape
    combined = np.zeros((h, w), dtype=np.uint8)

    for m in masks:
        if m is None:
            continue
        # Ensure same size
        if m.shape[:2] != (h, w):
            m = cv2.resize(m.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        combined = np.maximum(combined, (m > 0).astype(np.uint8) * 255)

    return combined


def calculate_mask_overlap(
    bbox: List[int],
    footpath_mask: Optional[np.ndarray],
    threshold: float = None,
) -> Dict[str, Any]:
    """
    Compute how much of the object bbox lies on the footpath mask.

    Returns dict with:
        intersection_pixels, bbox_area, overlap_ratio, is_encroachment
    """
    if threshold is None:
        threshold = settings.FOOTPATH_OVERLAP_THRESHOLD

    if footpath_mask is None:
        return {
            "intersection_pixels": 0,
            "bbox_area": 0,
            "overlap_ratio": 0.0,
            "is_encroachment": False,
        }

    h, w = footpath_mask.shape[:2]
    x1, y1, x2, y2 = bbox

    # Clip to image bounds
    x1 = max(0, min(w, int(x1)))
    y1 = max(0, min(h, int(y1)))
    x2 = max(0, min(w, int(x2)))
    y2 = max(0, min(h, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return {
            "intersection_pixels": 0,
            "bbox_area": 0,
            "overlap_ratio": 0.0,
            "is_encroachment": False,
        }

    crop = footpath_mask[y1:y2, x1:x2]
    intersection = int(np.count_nonzero(crop))
    area = (x2 - x1) * (y2 - y1)
    ratio = intersection / area if area > 0 else 0.0

    return {
        "intersection_pixels": intersection,
        "bbox_area": area,
        "overlap_ratio": ratio,
        "is_encroachment": ratio >= threshold,
    }


def is_inside_footpath(
    bbox: List[int],
    footpath_mask: Optional[np.ndarray],
    threshold: float = None,
) -> bool:
    return calculate_mask_overlap(bbox, footpath_mask, threshold)["is_encroachment"]


def filter_duplicates(
    detections: List[Dict[str, Any]],
    iou_threshold: float = None,
) -> List[Dict[str, Any]]:
    """
    Simple greedy NMS-style duplicate removal.
    Keeps higher-confidence detection when IoU exceeds threshold.
    """
    if iou_threshold is None:
        iou_threshold = settings.DUPLICATE_IOU_THRESHOLD

    if not detections:
        return []

    # Sort by confidence descending
    dets = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    keep = []

    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [
            d for d in dets
            if calculate_iou(best["bbox"], d["bbox"]) < iou_threshold
        ]

    return keep


# ------------------------------------------------------------------
# Drawing helpers
# ------------------------------------------------------------------

def draw_label(
    frame: np.ndarray,
    text: str,
    origin: Tuple[int, int],
    bg_color: Tuple[int, int, int],
    text_color: Tuple[int, int, int] = None,
    font_scale: float = None,
) -> None:
    if text_color is None:
        text_color = settings.TEXT_COLOR
    if font_scale is None:
        font_scale = settings.FONT_SCALE

    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, settings.FONT_THICKNESS
    )
    x, y = origin
    cv2.rectangle(
        frame,
        (x, y - th - baseline - 4),
        (x + tw + 4, y),
        bg_color,
        -1,
    )
    cv2.putText(
        frame,
        text,
        (x + 2, y - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        settings.FONT_THICKNESS,
        cv2.LINE_AA,
    )


def draw_bbox(
    frame: np.ndarray,
    det: Dict[str, Any],
) -> None:
    """
    Draw one detection with status-aware colouring.
    Expected keys in det:
        bbox, class_name, confidence, overlap_ratio, status
        status ∈ {"ENCROACHMENT", "NOT_ENCROACHMENT", "IGNORED"}
    """
    x1, y1, x2, y2 = det["bbox"]
    status = det.get("status", "NOT_ENCROACHMENT")
    name = det.get("class_name", "obj")
    conf = det.get("confidence", 0.0)
    overlap = det.get("overlap_ratio", 0.0)

    if status == "ENCROACHMENT":
        color = settings.COLOR_ENCROACHMENT
        status_text = "ENCROACHMENT"
    elif status == "IGNORED":
        color = settings.COLOR_IGNORED
        status_text = "IGNORED"
    else:
        color = settings.COLOR_NOT_ENCROACHMENT
        status_text = "OK"

    # Box
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, settings.BOX_THICKNESS)

    # Multi-line label
    lines = [
        f"{name} {conf:.2f}",
        f"Overlap: {overlap * 100:.1f}%",
        status_text,
    ]
    y_off = y1
    for i, line in enumerate(lines):
        draw_label(frame, line, (x1, y_off), color)
        y_off -= 18  # stack upward


def overlay_footpath_mask(
    frame: np.ndarray,
    footpath_mask: Optional[np.ndarray],
) -> np.ndarray:
    """Semi-transparent green overlay of the footpath mask."""
    if footpath_mask is None:
        return frame

    out = frame.copy()
    color_mask = np.zeros_like(frame)
    color_mask[footpath_mask > 0] = settings.COLOR_FOOTPATH_MASK

    alpha = settings.FOOTPATH_MASK_ALPHA
    mask_bool = footpath_mask > 0
    out[mask_bool] = (
        (1 - alpha) * out[mask_bool] + alpha * color_mask[mask_bool]
    ).astype(np.uint8)

    return out