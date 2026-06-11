import cv2
import time
from ultralytics import YOLO

# ---------------- CONFIGURATION ----------------
# Ensure this matches your actual path
MODEL_PATH = 'runs/detect/runs/detect/droneRTX/weights/best.pt' 
CAMERA_INDEX = 0  # 0 for internal webcam, 1 for USB
CONF_THRESHOLD = 0.5
# -----------------------------------------------

# Initialize Model
print(f"🚀 Loading DroneHawk (YOLO26m): {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

# Setup Camera
cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("❌ Error: Camera not found. Try changing CAMERA_INDEX to 1.")
    exit()

# Get dimensions for drawing the HUD
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print("✅ SYSTEM ONLINE. Press 'Q' to exit.")

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret: 
        break

    # 1. Inference
    results = model.predict(frame, conf=CONF_THRESHOLD, verbose=False)
    annotated_frame = results[0].plot()

    # 2. Calculate FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    # 3. Draw The "Military HUD"
    center_x, center_y = width // 2, height // 2
    color = (0, 255, 0) # Hacker Green

    # Crosshair
    cv2.line(annotated_frame, (center_x - 20, center_y), (center_x + 20, center_y), color, 2)
    cv2.line(annotated_frame, (center_x, center_y - 20), (center_x, center_y + 20), color, 2)
    
    # Stats Text
    cv2.putText(annotated_frame, f"MODEL: YOLO26m", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(annotated_frame, f"GPU: RTX 4060 | FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    cv2.putText(annotated_frame, "STATUS: SEARCHING", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Corner brackets (Viewfinder look)
    cv2.line(annotated_frame, (20, 20), (60, 20), color, 2)
    cv2.line(annotated_frame, (20, 20), (20, 60), color, 2)
    cv2.line(annotated_frame, (width-20, 20), (width-60, 20), color, 2)
    cv2.line(annotated_frame, (width-20, 20), (width-20, 60), color, 2)

    # 4. Show Output
    cv2.imshow('DroneHawk Defense System - LIVE', annotated_frame)

    # Quit on 'Q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()