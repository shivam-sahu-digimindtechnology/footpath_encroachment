

from pathlib import Path
from typing import List
import torch
from pydantic_settings import BaseSettings
from ultralytics import YOLO


# Project root = parent of app/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    FOOTPATH_MODEL_PATH: str = str(PROJECT_ROOT / "model" / "best (17).pt")
    STALL_MODEL_PATH: str = str(PROJECT_ROOT / "model" / "stalls.pt")
    DEBRIS_MODEL_PATH: str = str(PROJECT_ROOT / "model" / "debries.pt")
    OCCUPANCY_MODEL_PATH: str = str(PROJECT_ROOT / "model" / "occupied.pt")
    YOLO_MODEL_PATH: str = str(PROJECT_ROOT / "model" / "yolov8n.pt")
    # YOLO_MODEL_PATH: YOLO = YOLO("yolov8n.pt") 

    # INPUT_IMAGE: str = str(PROJECT_ROOT / "data" / "images" / "image.png")
    INPUT_VIDEO: str = str(PROJECT_ROOT / "data" / "video" / "FOOTPATH2.mp4")
    OUTPUT_DIR: str = str(PROJECT_ROOT / "output")
    OUTPUT_PATH: str = str(PROJECT_ROOT / "output" / "result.mp4")

    # ------------------------------------------------------------------
    # Runtime mode
    # ------------------------------------------------------------------
    USE_VIDEO: bool = True          # True = video, False = single image
    SHOW_WINDOW: bool = True
    SHOW_FPS: bool = True
    SAVE_OUTPUT: bool = True
    PRINT_DETECTIONS: bool = True

    # ------------------------------------------------------------------
    # Device
    # ------------------------------------------------------------------
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Confidence thresholds
    # ------------------------------------------------------------------
    FOOTPATH_CONFIDENCE: float = 0.35
    CONFIDENCE_THRESHOLD: float = 0.40   # for stalls / debris / occupancy / yolo

    # ------------------------------------------------------------------
    # Encroachment logic
    # ------------------------------------------------------------------
    # Even a small portion of the object on the footpath counts
    FOOTPATH_OVERLAP_THRESHOLD: float = 0   # 5%

    # IoU above which two detections are treated as the same object
    DUPLICATE_IOU_THRESHOLD: float = 0.50

    # ------------------------------------------------------------------
    # Classes that must NEVER be marked as encroachment
    # ------------------------------------------------------------------
    IGNORE_CLASSES: List[str] = [
        "person",
        "dog",
        "bus","truck","car","train","boat",
    ]

    INFRASTRUCTURE_CLASSES: List[str] = [
        "tree",
        "pole",
        "street light",
        "traffic light",
        "stop sign",
        "fire hydrant",
        "parking meter",
        "bench",          # keep if you consider fixed benches infrastructure
    ]

    # ------------------------------------------------------------------
    # Visualization (BGR)
    # ------------------------------------------------------------------
    COLOR_FOOTPATH_MASK: tuple = (0, 200, 0)       # green
    FOOTPATH_MASK_ALPHA: float = 0.35              # 0=invisible, 1=opaque

    COLOR_ENCROACHMENT: tuple = (0, 0, 255)        # red
    COLOR_NOT_ENCROACHMENT: tuple = (0, 255, 0)    # green
    COLOR_IGNORED: tuple = (180, 180, 180)         # gray
    COLOR_FOOTPATH_BOX: tuple = (255, 200, 0)      # cyan-ish (debug)

    TEXT_COLOR: tuple = (255, 255, 255)
    FONT_SCALE: float = 0.55
    BOX_THICKNESS: int = 2
    FONT_THICKNESS: int = 1


settings = Settings()