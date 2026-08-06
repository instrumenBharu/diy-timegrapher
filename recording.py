import os
from datetime import datetime

import sounddevice as sd
import soundfile as sf

# ==========================
# Configuration
# ==========================
SAMPLE_RATE = 44100      # Hz
DURATION = 3             # seconds
CHANNELS = 1             # Mono

OUTPUT_FOLDER = "wav_sample"

# Create folder if it doesn't exist
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Generate filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = os.path.join(
    OUTPUT_FOLDER,
    f"sample_{timestamp}.wav"
)

print("Recording...")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype="float32"
)

sd.wait()

print("Saving...")

sf.write(filename, audio, SAMPLE_RATE)

print(f"Done! Audio saved to:\n{filename}")