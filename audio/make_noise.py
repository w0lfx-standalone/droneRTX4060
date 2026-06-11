import numpy as np
import scipy.io.wavfile as wav

# Settings
SAMPLE_RATE = 22050
DURATION = 60  # 60 seconds of background audio
FILENAME = "dataset/background/synthetic_background.wav"

print("Generating synthetic background noise...")

# 1. Generate white noise (simulates wind/static)
# Amplitude of 0.01 makes it quiet (like a room)
noise = np.random.normal(0, 0.01, SAMPLE_RATE * DURATION)

# 2. Save as a standard WAV file
wav.write(FILENAME, SAMPLE_RATE, noise.astype(np.float32))

print(f"✅ Created valid file: {FILENAME}")