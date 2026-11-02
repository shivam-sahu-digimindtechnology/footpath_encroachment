"""
Model wrappers for Footpath Encroachment Detection.
All models are loaded once and reused.
"""

from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from ultralytics import YOLO
from config import settings


class FootpathDetector:
    """Instance-segmentation model that returns the Footpath mask only."""

    def __init__(self, model_path: str = settings.FOOTPATH_MODEL_PATH):
        self.model = YOLO(model_path)
        self.model.to(settings.DEVICE)
        print(f"[OK] Footpath model loaded → {model_path}  (device={settings.DEVICE})")

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Returns:
            {
              "mask": np.ndarray HxW uint8 (0/255) or None,
              "confidence": float,
              "boxes": list of footpath bboxes (debug),
            }
        """
        results = self.model(
            frame,
            conf=settings.FOOTPATH_CONFIDENCE,
            verbose=False,
            device=settings.DEVICE,
        )[0]

        h, w = frame.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        confidences = []
        boxes = []

        if results.masks is None or results.boxes is None:
            return {"mask": None, "confidence": 0.0, "boxes": []}

        names = results.names  # {id: name}

        for i, box in enumerate(results.boxes):
            cls_id = int(box.cls[0].item())
            label = names.get(cls_id, "").lower()

            # Only keep "Footpath", ignore "Crossing"
            if "footpath" not in label:
                continue

            conf = float(box.conf[0].item())
            confidences.append(conf)

            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
            boxes.append(xyxy)

            # Extract mask for this instance
            mask_tensor = results.masks.data[i]  # (Hm, Wm)
            mask_np = mask_tensor.cpu().numpy()

            # Resize to original frame size if needed
            if mask_np.shape[:2] != (h, w):
                import cv2
                mask_np = cv2.resize(
                    mask_np.astype(np.float32),
                    (w, h),
                    interpolation=cv2.INTER_LINEAR,
                )

            combined_mask = np.maximum(
                combined_mask,
                (mask_np > 0.5).astype(np.uint8) * 255,
            )

        avg_conf = float(np.mean(confidences)) if confidences else 0.0
        final_mask = combined_mask if np.any(combined_mask) else None

        return {
            "mask": final_mask,
            "confidence": avg_conf,
            "boxes": boxes,
        }


class ObjectDetector:
    """Generic wrapper for any detection YOLO model (stalls, debris, occupancy, yolov8n)."""

    def __init__(self, model_path: str, model_name: str):
        self.model = YOLO(model_path)
        self.model.to(settings.DEVICE)
        self.model_name = model_name
        print(f"[OK] {model_name} model loaded → {model_path}")

    def detect(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        results = self.model(
            frame,
            conf=settings.CONFIDENCE_THRESHOLD,
            verbose=False,
            device=settings.DEVICE,
        )[0]

        detections = []
        if results.boxes is None:
            return detections

        names = results.names

        for box in results.boxes:
            cls_id = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            label = names.get(cls_id, f"class_{cls_id}")
            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()

            detections.append({
                "model": self.model_name,
                "class_id": cls_id,
                "class_name": label,
                "confidence": conf,
                "bbox": xyxy,  # [x1, y1, x2, y2]
            })

        return detections


class MultiModelDetector:
    """Loads all object models and runs them in one call."""

    def __init__(self):
        self.stall_det = ObjectDetector(settings.STALL_MODEL_PATH, "stalls")
        self.debris_det = ObjectDetector(settings.DEBRIS_MODEL_PATH, "debris")
        self.occupancy_det = ObjectDetector(settings.OCCUPANCY_MODEL_PATH, "occupancy")
        self.yolo_det = ObjectDetector(settings.YOLO_MODEL_PATH, "yolov8")

    def detect_all(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        dets = []
        dets.extend(self.stall_det.detect(frame))
        dets.extend(self.debris_det.detect(frame))
        dets.extend(self.occupancy_det.detect(frame))
        dets.extend(self.yolo_det.detect(frame))
        return dets