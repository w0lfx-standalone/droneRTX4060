import os
import numpy as np
import scipy.io.wavfile as wav
import librosa
from tqdm import tqdm
import shutil

# --- Settings ---
BASE_DIR = os.getcwd()
DATASET_PATH = os.path.join(BASE_DIR, "dataset")
BG_FOLDER = os.path.join(DATASET_PATH, "background")
OUTPUT_PATH = os.path.join(BASE_DIR, "processed_data")

SAMPLE_RATE = 22050
DURATION = 2.0
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)

def create_synthetic_noise():
    print("--- 1. Fixing Background Data ---")
    
    # 1. Clean the folder (Delete bad files)
    if os.path.exists(BG_FOLDER):
        for filename in os.listdir(BG_FOLDER):
            file_path = os.path.join(BG_FOLDER, filename)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Could not delete {filename}: {e}")
    else:
        os.makedirs(BG_FOLDER)

    # 2. Generate clean White Noise (1 minute long)
    print("Generating clean synthetic noise...")
    noise_data = np.random.normal(0, 0.005, SAMPLE_RATE * 60) # Low volume static
    
    # 3. Save as standard WAV
    noise_path = os.path.join(BG_FOLDER, "synthetic_noise.wav")
    wav.write(noise_path, SAMPLE_RATE, noise_data.astype(np.float32))
    print(f"✅ Created valid background file: {noise_path}")

def save_spectrogram(file_path, label):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Split into chunks
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
            
            filename = os.path.basename(file_path).replace('.wav', '') + f'_{i}.npy'
            target_path = os.path.join(OUTPUT_PATH, label)
            os.makedirs(target_path, exist_ok=True)
            np.save(os.path.join(target_path, filename), mel_spec_norm)
            
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")

def run_processing():
    print("\n--- 2. Processing Data ---")
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    for label in ['drone', 'background']:
        folder_path = os.path.join(DATASET_PATH, label)
        files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]
        
        print(f"Processing {label} ({len(files)} files)...")
        for f in tqdm(files):
            save_spectrogram(os.path.join(folder_path, f), label)

if __name__ == "__main__":
    # Ensure scipy is installed
    try:
        import scipy
    except ImportError:
        print("Installing missing library (scipy)...")
        os.system("pip install scipy")

    create_synthetic_noise()
    run_processing()
    print("\n--- ✅ DONE! You can now run 'python train.py' ---")