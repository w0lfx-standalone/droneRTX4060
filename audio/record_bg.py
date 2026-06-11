import sounddevice as sd
import scipy.io.wavfile as wav
import os

# --- Settings ---
DURATION = 300  # Record 30 seconds of "Silence"
SAMPLE_RATE = 22050
FILENAME = "dataset/background/real_room_noise.wav"

print(f"--- 🎙️ RECORDING BACKGROUND ({DURATION}s) ---")
print("1. Please stay quiet.")
print("2. Let the mic capture your room's natural hum (fans, AC, wind).")
print("Recording starts in 3 seconds...")
sd.sleep(3000)

print("🔴 RECORDING...")
# Record audio
recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
sd.wait()  # Wait until recording is finished

# Save to file
os.makedirs(os.path.dirname(FILENAME), exist_ok=True)
wav.write(FILENAME, SAMPLE_RATE, recording)

print(f"✅ Saved! File located at: {FILENAME}")
print("You must now re-run 'master_setup.py' to process this new file.")