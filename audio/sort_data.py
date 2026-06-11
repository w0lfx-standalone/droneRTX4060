import os
import shutil

# Define your source and destination
current_folder = os.getcwd()
dataset_root = os.path.join(current_folder, "dataset")
drone_folder = os.path.join(dataset_root, "drone")
background_folder = os.path.join(dataset_root, "background")
unused_folder = os.path.join(dataset_root, "unused_calibration")

# Create directories
for folder in [drone_folder, background_folder, unused_folder]:
    os.makedirs(folder, exist_ok=True)

print("--- Sorting Files ---")

# Scan and move files
for filename in os.listdir(current_folder):
    if not filename.endswith(".wav"):
        continue
        
    src_path = os.path.join(current_folder, filename)
    
    # Logic: 'Ed_' files go to Drone, 'Calib' goes to unused
    if filename.startswith("Ed_"):
        shutil.move(src_path, os.path.join(drone_folder, filename))
        print(f"Moved to Drone: {filename}")
        
    elif filename.startswith("Calib"):
        shutil.move(src_path, os.path.join(unused_folder, filename))
        print(f"Moved to Unused: {filename}")

print("\n--- Sorting Complete ---")
print(f"1. Drone files are in: {drone_folder}")
print(f"2. Calibration files moved to: {unused_folder}")
print(f"3. IMPORTANT: The '{background_folder}' is currently EMPTY.")