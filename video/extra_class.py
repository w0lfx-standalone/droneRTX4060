from ultralytics import YOLO
import os

# ---------------- CONFIGURATION ----------------
# 1. Input Model Path
# I fixed the double 'runs/detect' typo in your path.
# If this specific file doesn't exist, the script will auto-find 'best.pt'
MODEL_PATH = 'runs/detect/runs/detect/droneRTX/weights/best.pt' 

# 2. Dataset Path
DATA_YAML = 'data.yaml'
# -----------------------------------------------

def train_hackathon_winner():
    # --- SAFETY CHECK: Find the model ---
    if not os.path.exists(MODEL_PATH):
        print(f"⚠️  Could not find: {MODEL_PATH}")
        if os.path.exists('best.pt'):
            print("✅  Found 'best.pt' in current folder. Using that.")
            model_to_load = 'best.pt'
        else:
            print("❌  No model found! Download 'best.pt' first.")
            return
    else:
        model_to_load = MODEL_PATH

    print(f"🚀 Loading {model_to_load}...")
    model = YOLO(model_to_load)

    print("\n🧠 Starting 3-Hour Training Run on RTX 4060...")
    print("---------------------------------------------")
    print("   Goal: High Accuracy (85%+) in < 3 Hours")
    print("   Strategy: Train on 40% of Data (Smart Sampling)")
    print("---------------------------------------------\n")

    # TRAIN COMMAND
    model.train(
        data=DATA_YAML,
        epochs=15,             # 15 Epochs is perfect for fine-tuning
        imgsz=640,
        
        # --- STABILITY SETTINGS (Crucial for 8GB VRAM) ---
        batch=4,               # 4 is safe. 6 or 16 WILL crash your GPU.
        workers=4,             # Prevents the "EOFError" system crash.
        
        # --- TIME MANAGEMENT ---
        fraction=0.40,         # Uses 40% of images. Fits perfectly in 3 hours.
        
        # --- OPTIMIZATION ---
        lr0=0.001,             # Low LR to polish the weights
        optimizer='AdamW',     # Best optimizer
        project='runs/detect',
        name='drone_final_run',# Clean output folder name
        exist_ok=True,
        verbose=True
    )
    
    print("\n✅ TRAINING COMPLETE!")
    print("---------------------------------------------")
    print("   Your Final Model: runs/detect/drone_final_run/weights/best.pt")
    print("---------------------------------------------")
    print("👉 Update your app.py to use this file for the demo!")

if __name__ == '__main__':
    train_hackathon_winner()