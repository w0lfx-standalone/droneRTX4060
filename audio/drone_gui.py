import tkinter as tk
from tkinter import ttk
import sounddevice as sd
import numpy as np
import librosa
import torch
import torch.nn as nn
import threading
import queue

# --- Configuration ---
MODEL_PATH = "drone_catcher_model.pth"
SAMPLE_RATE = 22050
DURATION = 2.0 
THRESHOLD = 0.8 # 80% confidence

# --- 1. The Brain (Must match your trained model) ---
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

# --- 2. The GUI Application ---
class DroneCatcherUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🦅 Drone Noise Catcher (GTX 1650)")
        self.root.geometry("500x350")
        self.root.configure(bg="#1e1e1e") # Dark mode

        self.is_running = False
        self.queue = queue.Queue()

        # Load AI
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DroneNet().to(self.device)
        try:
            self.model.load_state_dict(torch.load(MODEL_PATH))
            self.model.eval()
            print(f"✅ AI Loaded on {self.device}")
        except:
            print("❌ Error: Model not found! Run train.py first.")

        # --- Layout ---
        # Title
        tk.Label(root, text="Acoustic Drone Radar", font=("Arial", 18, "bold"), 
                 bg="#1e1e1e", fg="white").pack(pady=20)

        # Status Indicator
        self.status_label = tk.Label(root, text="SAFE", font=("Arial", 24, "bold"), 
                                     bg="#1e1e1e", fg="#00ff00") # Green = Safe
        self.status_label.pack(pady=10)

        # Confidence Bar
        self.progress = ttk.Progressbar(root, orient="horizontal", length=400, mode="determinate")
        self.progress.pack(pady=20)
        
        # Percentage Label
        self.percent_label = tk.Label(root, text="0%", font=("Arial", 12), bg="#1e1e1e", fg="gray")
        self.percent_label.pack()

        # Buttons
        btn_frame = tk.Frame(root, bg="#1e1e1e")
        btn_frame.pack(pady=30)

        self.btn_start = tk.Button(btn_frame, text="START SCANNING", font=("Arial", 12, "bold"), 
                                   bg="#007acc", fg="white", command=self.start_listening)
        self.btn_start.pack(side=tk.LEFT, padx=10)

        self.btn_stop = tk.Button(btn_frame, text="STOP", font=("Arial", 12, "bold"), 
                                  bg="#cc0000", fg="white", command=self.stop_listening, state=tk.DISABLED)
        self.btn_stop.pack(side=tk.LEFT, padx=10)

        # Start UI Update Loop
        self.root.after(100, self.update_ui)

    def audio_callback(self, indata, frames, time, status):
        """Runs in background thread"""
        if self.is_running:
            # Prepare data for AI
            audio_data = indata.flatten()
            mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=SAMPLE_RATE, n_mels=64)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
            mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-6)
            
            # Send to GPU
            tensor = torch.tensor(mel_spec_norm, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(tensor)
                prob = torch.softmax(output, dim=1)
                drone_conf = prob[0][1].item() # Probability of Drone
            
            # Send result to UI
            self.queue.put(drone_conf)

    def start_listening(self):
        self.is_running = True
        self.btn_start.config(state=tk.DISABLED, bg="gray")
        self.btn_stop.config(state=tk.NORMAL, bg="#cc0000")
        
        # Start Audio Stream in separate thread
        self.stream_thread = threading.Thread(target=self.run_stream)
        self.stream_thread.start()

    def run_stream(self):
        with sd.InputStream(callback=self.audio_callback, channels=1, samplerate=SAMPLE_RATE, 
                            blocksize=int(SAMPLE_RATE * DURATION)):
            while self.is_running:
                sd.sleep(100)

    def stop_listening(self):
        self.is_running = False
        self.btn_start.config(state=tk.NORMAL, bg="#007acc")
        self.btn_stop.config(state=tk.DISABLED, bg="gray")
        self.status_label.config(text="PAUSED", fg="gray")
        self.progress["value"] = 0
        self.percent_label.config(text="0%")

    def update_ui(self):
        """Checks for new predictions and updates the window"""
        try:
            while not self.queue.empty():
                confidence = self.queue.get_nowait()
                
                # Update Bar
                percentage = int(confidence * 100)
                self.progress["value"] = percentage
                self.percent_label.config(text=f"{percentage}% Drone Confidence")

                # Update Status Text & Color
                if confidence > THRESHOLD:
                    self.status_label.config(text="🚨 DRONE DETECTED!", fg="#ff0000") # Red
                else:
                    self.status_label.config(text="SAFE (Scanning...)", fg="#00ff00") # Green
        except:
            pass
        
        self.root.after(100, self.update_ui)

if __name__ == "__main__":
    root = tk.Tk()
    app = DroneCatcherUI(root)
    root.mainloop()