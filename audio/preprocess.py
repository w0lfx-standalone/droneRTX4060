import os
import librosa
import numpy as np
from tqdm import tqdm

# --- Configuration ---
# We use 'os.getcwd()' to make sure it finds the folders relative to where you run the script
BASE_DIR = os.getcwd()
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
OUTPUT_PATH = os.path.join(BASE_DIR, "processed_data")

SAMPLE_RATE = 22050
DURATION = 2.0  # Seconds per chunk
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)

def save_spectrogram(file_path, label, save_dir):
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Split into 2-second chunks
        num_chunks = int(len(y) / SAMPLES_PER_TRACK)
        
        # If the file is shorter than 2 seconds, pad it
        if num_chunks == 0:
             y = librosa.util.fix_length(y, size=SAMPLES_PER_TRACK)
             num_chunks = 1

        for i in range(num_chunks):
            start = i * SAMPLES_PER_TRACK
            end = start + SAMPLES_PER_TRACK
            chunk = y[start:end]
            
            # Create Mel Spectrogram (The "Image" of the sound)
            mel_spec = librosa.feature.melspectrogram(y=chunk, sr=sr, n_mels=64)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            
            # Normalize to 0-1 range (Critical for the Neural Network)
            mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-6)
            
            # Save as NumPy file
            filename = os.path.basename(file_path).replace('.wav', f'_{i}.npy')
            target_path = os.path.join(save_dir, label)
            os.makedirs(target_path, exist_ok=True)
            np.save(os.path.join(target_path, filename), mel_spec_norm)
            
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def preprocess_dataset():
    classes = ['drone', 'background']
    
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    print(f"Looking for data in: {DATASET_PATH}")

    for label in classes:
        folder_path = os.path.join(DATASET_PATH, label)
        
        # Check if folder exists and has files
        if not os.path.exists(folder_path) or not os.listdir(folder_path):
            print(f"❌ ERROR: '{label}' folder is missing or empty! ({folder_path})")
            continue
            
        print(f"Processing {label}...")
        files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        
        # tqdm shows a progress bar
        for f in tqdm(files):
            save_spectrogram(os.path.join(folder_path, f), label, OUTPUT_PATH)

if __name__ == "__main__":
    print("--- Starting Audio Processor ---")
    preprocess_dataset()
    print("--- Done! Data ready for GTX 1650 ---")