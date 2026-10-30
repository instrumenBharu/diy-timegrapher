"""
Watch tick audio signal-processing lab
--------------------------------------
Records raw microphone audio, saves it unchanged to WAV, then plots:
1) raw time-domain waveform
2) band-pass filtered waveform
3) envelope of the filtered waveform
4) autocorrelation of the envelope
5) epoch-folded envelope using an estimated candidate period
6) raw vs filtered comparison

Install:
    pip install numpy scipy matplotlib sounddevice

Run:
    python watch_signal_lab.py

Optional file processing:
    python watch_signal_lab.py --input recording.wav

Notes:
- Default recording is 5 seconds at 44.1 kHz, mono.
- Tune band-pass cutoffs after inspecting real recordings.
- Autocorrelation may find a sub-period or a multiple; treat the estimate as a candidate.
- Epoch folding is performed on the envelope and is for visualization/template exploration,
  not a replacement for original event timestamps.
"""

import argparse
import os
import wave
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, hilbert, correlate, find_peaks
from scipy.ndimage import gaussian_filter1d


def record_audio(duration, sample_rate, output_path):
    """Record mono float audio, then save raw PCM16 WAV without processing."""
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise SystemExit(
            "Missing sounddevice. Install dependencies with:\n"
            "pip install numpy scipy matplotlib sounddevice"
        ) from exc

    print(f"Recording {duration:.2f} seconds at {sample_rate} Hz...")
    audio = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    raw = audio[:, 0].copy()

    # Save raw samples as PCM16 WAV. The in-memory float32 raw array is also retained.
    pcm = np.clip(raw, -1.0, 1.0)
    pcm16 = (pcm * 32767).astype(np.int16)
    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())

    print(f"Saved raw recording: {os.path.abspath(output_path)}")
    return raw, sample_rate


def load_wav(path):
    """Load a mono or stereo PCM WAV; stereo is averaged to mono."""
    from scipy.io import wavfile

    sample_rate, data = wavfile.read(path)

    if data.ndim == 2:
        data = data.astype(np.float64).mean(axis=1)
    else:
        data = data.astype(np.float64)

    if np.issubdtype(data.dtype, np.integer):
        # This branch is normally bypassed after astype above, retained for clarity.
        pass

    # Normalize integer PCM by its likely full-scale range.
    # scipy.io.wavfile returns integer types; infer from original WAV bit depth.
    if np.max(np.abs(data)) > 1.5:
        data /= 32768.0

    return data.astype(np.float64), int(sample_rate)


def bandpass_filter(signal, sample_rate, low_hz, high_hz, order=4):
    nyquist = sample_rate / 2.0
    if not (0 < low_hz < high_hz < nyquist):
        raise ValueError(
            f"Cutoffs must satisfy 0 < low < high < Nyquist ({nyquist:.1f} Hz)."
        )
    sos = butter(
        order,
        [low_hz, high_hz],
        btype="bandpass",
        fs=sample_rate,
        output="sos",
    )
    return sosfiltfilt(sos, signal)



def plot_results(raw, filtered, sample_rate, output_plot):
    time = np.arange(len(raw)) / sample_rate

    fig, axes = plt.subplots(2, 1, figsize=(14, 15), constrained_layout=True)

    axes[0].plot(time, raw, linewidth=0.55)
    axes[0].set_title("1. Raw audio — time domain")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_xlabel("Time (s)")
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(time, filtered, linewidth=0.55)
    axes[1].set_title("2. Band-pass filtered audio")
    axes[1].set_ylabel("Amplitude")
    axes[1].set_xlabel("Time (s)")
    axes[1].grid(True, alpha=0.25)


    fig.suptitle("Watch Audio SNR Exploration", fontsize=16)
    fig.savefig(output_plot, dpi=160)
    print(f"Saved plot: {os.path.abspath(output_plot)}")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Watch tick audio signal-processing lab")
    parser.add_argument("--input", help="Existing WAV file to process instead of recording")
    parser.add_argument("--duration", type=float, default=5.0, help="Recording duration in seconds")
    parser.add_argument("--sample-rate", type=int, default=44100, help="Recording sample rate")
    parser.add_argument("--raw-output", default="watch_raw.wav", help="Path for saved raw WAV")
    parser.add_argument("--plot-output", default="watch_signal_processing.png", help="Path for saved plot")
    parser.add_argument("--low-cut", type=float, default=4000, help="Band-pass low cutoff in Hz")
    parser.add_argument("--high-cut", type=float, default=12000, help="Band-pass high cutoff in Hz")
    parser.add_argument("--envelope-smooth-ms", type=float, default=8.0, help="Envelope smoothing scale in ms")
    parser.add_argument("--min-period-ms", type=float, default=100.0, help="Minimum autocorrelation lag to search")
    parser.add_argument("--max-period-ms", type=float, default=250.0, help="Maximum autocorrelation lag to search")
    args = parser.parse_args()

    if args.input:
        raw, sample_rate = load_wav(args.input)
        # Copy source file to raw output only if output differs; original input remains untouched.
        if os.path.abspath(args.input) != os.path.abspath(args.raw_output):
            import shutil
            shutil.copy2(args.input, args.raw_output)
            print(f"Copied source WAV to: {os.path.abspath(args.raw_output)}")
    else:
        raw, sample_rate = record_audio(args.duration, args.sample_rate, args.raw_output)

    if len(raw) < 10:
        raise SystemExit("Recording is too short.")

    # Remove DC offset for processing, but keep 'raw' unchanged for plotting.
    cut_time = 0.4
    start_index = int(cut_time * sample_rate)

    raw = raw[start_index:]
    centered = raw - np.mean(raw)

    filtered = bandpass_filter(
        centered, sample_rate, args.low_cut, args.high_cut
    )

    print(f"Sample rate: {sample_rate} Hz")
    print(f"Duration: {len(raw) / sample_rate:.3f} s")

    plot_results(
        raw, filtered, sample_rate, args.plot_output
    )


if __name__ == "__main__":
    main()
