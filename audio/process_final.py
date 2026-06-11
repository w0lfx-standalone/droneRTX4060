import os
import shutil
import numpy as np
import librosa
from tqdm import tqdm

# Settings
BASE_DIR = os.getcwd()
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
OUTPUT_PATH = os.path.join(BASE_DIR, "processed_data")
SAMPLE_RATE = 22050
DURATION = 2.0 
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)

def save_spectrogram(file_path, label):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        # Pad if too short
        if len(y) < SAMPLES_PER_TRACK:
             y = librosa.util.fix_length(y, size=SAMPLES_PER_TRACK)

        num_chunks = int(len(y) / SAMPLES_PER_TRACK)
        
        for i in range(num_chunks):
            start = i * SAMPLES_PER_TRACK
            end = start + SAMPLES_PER_TRACK
            chunk = y[start:end]
            
            mel_spec = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=64)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-6)
            
            filename = os.path.basename(file_path).replace('.wav', '') + f'_{i}.npy'
            target_path = os.path.join(OUTPUT_PATH, label)
            os.makedirs(target_path, exist_ok=True)
            np.save(os.path.join(target_path, filename), mel_spec_norm)
    except Exception as e:
        print(f"Error: {e}")

def run():
    # 1. Clear old data to ensure we use ONLY the new recording
    if os.path.exists(OUTPUT_PATH):
        shutil.rmtree(OUTPUT_PATH)
        print("Cleared old processed data.")
    os.makedirs(OUTPUT_PATH)

    # 2. Process Drone and Background
    for label in ['drone', 'background']:
        folder = os.path.join(DATASET_PATH, label)
        if not os.path.exists(folder):
            print(f"Skipping {label} (Folder missing)")
            continue
            
        files = [f for f in os.listdir(folder) if f.endswith('.wav')]
        print(f"Processing {label} ({len(files)} files)...")
        for f in tqdm(files):
            save_spectrogram(os.path.join(folder, f), label)

if __name__ == "__main__":
    run()