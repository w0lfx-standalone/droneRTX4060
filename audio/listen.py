import sounddevice as sd
import numpy as np
import librosa
import torch
import torch.nn as nn
import time
import os

# --- Configuration ---
MODEL_PATH = "drone_catcher_model.pth"
SAMPLE_RATE = 22050
DURATION = 2.0  # Seconds the AI listens to at once
THRESHOLD = 0.8 # 80% confidence required to trigger alert

# --- 1. Re-define the Neural Network (Must match training!) ---
class DroneNet(nn.Module):
    def __init__(self):
        super(DroneNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.global_pool = nn.AdaptiveAvgPool2d((4, 4)) 
        self.fc1 = nn.Linear(64 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = self.global_pool(x)
        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 2. Helper Functions ---
def audio_to_spectrogram(audio_buffer):
    # Create Mel Spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio_buffer, sr=SAMPLE_RATE, n_mels=64)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize (0-1)
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-6)
    
    # Convert to Tensor (1, 1, 64, Time)
    spec_tensor = torch.tensor(mel_spec_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return spec_tensor

def main():
    print("--- Loading Brain (Model) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DroneNet().to(device)
    
    if not os.path.exists(MODEL_PATH):
        print("❌ ERROR: Model file not found! Did you run train.py?")
        return

    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()
    print(f"✅ Model Loaded on {device}")
    print("\n--- 🎧 LISTENING FOR DRONES... (Press Ctrl+C to Stop) ---")

    # Audio Buffer setup
    buffer_size = int(SAMPLE_RATE * DURATION)
    
    def callback(indata, frames, time, status):
        """This function runs constantly in the background"""
        if status:
            print(status)
        
        # 1. Grab audio from mic
        audio_data = indata.flatten()
        
        # 2. Process
        with torch.no_grad():
            input_tensor = audio_to_spectrogram(audio_data).to(device)
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            drone_conf = probabilities[0][1].item() # Confidence it is a drone
        
        # 3. Visual Output
        bar_length = 20
        filled_length = int(bar_length * drone_conf)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)
        
        if drone_conf > THRESHOLD:
            print(f"\r🚨 DRONE DETECTED! [{bar}] {drone_conf*100:.1f}%   ", end="")
        else:
            print(f"\r   Scanning...     [{bar}] {drone_conf*100:.1f}%   ", end="")

    # Start the stream
    try:
        with sd.InputStream(callback=callback, channels=1, samplerate=SAMPLE_RATE, blocksize=buffer_size):
            while True:
                sd.sleep(100) # Keep script alive
    except KeyboardInterrupt:
        print("\n\n🛑 Stopped.")
    except Exception as e:
        print(f"\n❌ Microphone Error: {e}")

if __name__ == "__main__":
    main()