import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt

SAMPLE_RATE = 44100       
DURATION = 4             
CHANNELS = 1              

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


audio = audio[:, 0]
print(f"Audio converted to 1D array with {audio.shape} samples.")

time = np.arange(len(audio)) / SAMPLE_RATE
print (time.shape)


N = len(audio)

fft_result = np.fft.rfft(audio) #y axis

frequencies = np.fft.rfftfreq(N, d=1 / SAMPLE_RATE) #x axis

magnitude = np.abs(fft_result) #only positive y

plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)

plt.plot(time, audio)

# print(time)
# print(audio)

plt.title("Time - No Filter")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude")
plt.grid(True)

# ---- Frequency Domain ----
plt.subplot(2, 1, 2)

plt.plot(frequencies, magnitude)
plt.xlim(0, 100)

# print(frequencies)
# print(magnitude)
plt.title("Frequency - No Filter")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.grid(True)

plt.tight_layout()
plt.show()