import os
import librosa
import numpy as np
import shutil
from tqdm import tqdm

# --- Settings ---
BASE_DIR = os.getcwd()
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
OUTPUT_PATH = os.path.join(BASE_DIR, "processed_data")
SAMPLE_RATE = 22050
DURATION = 2.0 
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)

def fix_background_file():
    bg_folder = os.path.join(DATASET_PATH, "background")
    files = [f for f in os.listdir(bg_folder) if not f.startswith('.')] # Ignore hidden files
    
    if len(files) == 0:
        print("❌ CRITICAL ERROR: The background folder is EMPTY.")
        print(f"Please put your recording inside: {bg_folder}")
        return False
    
    # Grab the first file we find, whatever it is named
    current_file = files[0]
    current_path = os.path.join(bg_folder, current_file)
    
    # Force rename it to 'fixed_background.wav' so we know it's correct
    new_path = os.path.join(bg_folder, "fixed_background.wav")
    
    if current_file != "fixed_background.wav":
        try:
            os.rename(current_path, new_path)
            print(f"✅ FIXED: Renamed '{current_file}' to 'fixed_background.wav'")
        except:
            print(f"⚠️ Could not rename {current_file}, trying to read it anyway...")
            new_path = current_path

    return True

def save_spectrogram(file_path, label, save_dir):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        num_chunks = int(len(y) / SAMPLES_PER_TRACK)
        if num_chunks == 0:
             y = librosa.util.fix_length(y, size=SAMPLES_PER_TRACK)
             num_chunks = 1

        for i in range(num_chunks):
            start = i * SAMPLES_PER_TRACK
            end = start + SAMPLES_PER_TRACK
            chunk = y[start:end]
            mel_spec = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=64)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-6)
            
            filename = os.path.basename(file_path).replace('.wav', '').replace('.', '') + f'_{i}.npy'
            target_path = os.path.join(save_dir, label)
            os.makedirs(target_path, exist_ok=True)
            np.save(os.path.join(target_path, filename), mel_spec_norm)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def run_processing():
    if not fix_background_file():
        return

    classes = ['drone', 'background']
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    print("--- Processing Data ---")
    for label in classes:
        folder_path = os.path.join(DATASET_PATH, label)
        files = os.listdir(folder_path) # Read EVERYTHING, don't filter for .wav yet
        
        print(f"Processing {label} ({len(files)} files)...")
        for f in tqdm(files):
            # Process everything that isn't a hidden system file
            if not f.startswith('.'):
                save_spectrogram(os.path.join(folder_path, f), label, OUTPUT_PATH)

if __name__ == "__main__":
    run_processing()
    print("\n--- DONE! You can now run 'python train.py' ---")