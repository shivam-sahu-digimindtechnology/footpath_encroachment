import cv2
from ultralytics import YOLO

# ====================== CONFIG ======================
model_path = r"F:\footpath\detector\best (19).pt"
image_path = r"F:\footpath\detector\image4.webp"
output_path = r"F:\footpath\output\img.png"

SIDEWALK_CLASS_ID = 4          # change if your class id is different
CONFIDENCE = 0.5
# ====================================================

# Load model
model = YOLO(model_path)

# Run inference
results = model(image_path, conf=CONFIDENCE)
result = results[0]

sidewalk_found = False

# Check detections
if result.masks is not None:
    for i, cls in enumerate(result.boxes.cls):
        class_id = int(cls)
        class_name = model.names[class_id]
        conf = float(result.boxes.conf[i])

        print(f"Detected: {class_name} | Confidence: {conf:.2f}")

        if class_id == SIDEWALK_CLASS_ID:
            sidewalk_found = True
            print(f"✅ Sidewalk / Footpath detected | Confidence: {conf:.2f}")

# Final decision
print("\n" + "="*40)
if sidewalk_found:
    print("✅ FOOTPATH / SIDEWALK PRESENT")
else:
    print("❌ FOOTPATH / SIDEWALK NOT FOUND")
print("="*40)

# Save annotated image
annotated = result.plot()
cv2.imwrite(output_path, annotated)
print(f"\nAnnotated image saved → {output_path}")