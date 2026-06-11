from ultralytics import YOLO

# ---------------- CONFIGURATION ----------------
# ⚠️ MAKE SURE THIS POINTS TO THE DOWNLOADED MODEL
# Your logs showed you were pointing to the OLD model in 'runs/.../droneRTX'
# If you downloaded the new one, it is likely just 'best.pt' in the current folder.
MODEL_PATH = 'runs/detect/runs/detect/droneRTX/weights/best.pt' 

# Point to your data.yaml
DATA_YAML = 'data.yaml'
# -----------------------------------------------

print(f"🚀 Starting Validation on {MODEL_PATH}...")
model = YOLO(MODEL_PATH)

# --- THE FIX IS HERE ---
# single_cls=True: Treats all label classes (0,1,2) as just "Target". 
# This prevents the crash when the dataset has more classes than the model.
metrics = model.val(
    data=DATA_YAML, 
    split='test', 
    verbose=False, 
    single_cls=True   # <--- THIS STOPS THE CRASH
)

# Extract Key Metrics
accuracy = metrics.box.map50 * 100    # mAP50
precision = metrics.box.mp * 100      # Precision
recall = metrics.box.mr * 100         # Recall

print("\n" + "="*40)
print("   🏆 DRONEHAWK REPORT CARD 🏆")
print("="*40)
print(f"✅ Accuracy (mAP50):  {accuracy:.2f}%")
print(f"🎯 Precision:        {precision:.2f}%")
print(f"👀 Recall:           {recall:.2f}%")
print("="*40)

# Speed Check
if hasattr(metrics, 'speed'):
    speed = metrics.speed['inference']
    print(f"⚡ Speed: {speed:.2f} ms ({1000/speed:.0f} FPS)")
print("="*40 + "\n")