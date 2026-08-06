import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

# =========================
# SETTINGS
# =========================
SAMPLE_RATE = 44100       # Samples per second (Hz)
DURATION = 4              # Recording duration in seconds
CHANNELS = 1              # Mono microphone

# =========================
# RECORD AUDIO
# =========================
print("Recording...")

audio = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=CHANNELS,
    dtype="float32"
)

sd.wait()

print("Recording complete.")
print(f"Audio shape: {audio.shape}")

# Convert from (N, 1) to 1D array (N,)
audio = audio[:, 0]
print(f"Audio converted to 1D array with {audio.shape} samples.")
# =========================
# TIME DOMAIN
# =========================
time = np.arange(len(audio)) / SAMPLE_RATE
print (time.shape)
# =========================
# WIENER FILTER
# =========================



# =========================
# FREQUENCY DOMAIN
# =========================
N = len(audio)

# FFT
fft_result = np.fft.rfft(audio) #y axis

# Frequency axis
frequencies = np.fft.rfftfreq(N, d=1 / SAMPLE_RATE) #x axis

# Magnitude
magnitude = np.abs(fft_result) #only positive y

# =========================
# PLOT
# =========================
plt.figure(figsize=(12, 8))

# ---- Time Domain ----
plt.subplot(2, 1, 1)

plt.plot(time, audio)

# print(time)
# print(audio)

plt.title("Time - Wiener Filter")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)

# ---- Frequency Domain ----
plt.subplot(2, 1, 2)

plt.plot(frequencies, magnitude)
plt.xlim(0, 100)

# print(frequencies)
# print(magnitude)
plt.title("Frequency - Wiener Filter")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)

plt.tight_layout()
plt.show()