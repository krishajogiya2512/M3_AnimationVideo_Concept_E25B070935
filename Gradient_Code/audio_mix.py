"""Audio preparation and subtle procedural underscore mixer.

1. Normalizes and cleans Krisha's recorded voice-over (highpass + gentle denoise + loudness norm).
2. Synthesizes a subtle, warm ambient harmonic pad in NumPy/SciPy (clearly credited on Closing Frame).
3. Sidechain-ducks the ambient pad beneath speech and pads cleanly to TOTAL_DURATION.
"""

import os
import subprocess
import wave
import numpy as np
from timeline import TOTAL_DURATION

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_AUDIO = os.path.join(BASE_DIR, "assets", "voiceover.m4a")


def build_ambient_pad(duration_sec: float, sr: int = 44100) -> np.ndarray:
    """Synthesize a gentle, low-frequency warm ambient chord progression."""
    n = int(round(duration_sec * sr))
    t = np.linspace(0.0, duration_sec, n, endpoint=False)
    # Soft harmonic progression (Cadd9 -> Am9 -> Fmaj9 -> G6) cycling gently every 16s
    chords = [
        [130.81, 196.00, 293.66, 329.63],  # C3, G3, D4, E4
        [110.00, 164.81, 246.94, 329.63],  # A2, E3, B3, E4
        [174.61, 220.00, 261.63, 392.00],  # F3, A3, C4, G4
        [196.00, 246.94, 293.66, 329.63],  # G3, B3, D4, E4
    ]
    seg_len = 8.0
    sig = np.zeros(n, dtype=np.float64)
    # Smooth overlapping Hann windows per chord block
    num_blocks = int(np.ceil(duration_sec / seg_len)) + 1
    for b in range(num_blocks):
        t0 = b * seg_len - 1.5
        t1 = (b + 1) * seg_len + 1.5
        i0 = max(0, int(t0 * sr))
        i1 = min(n, int(t1 * sr))
        if i1 <= i0:
            continue
        tb = t[i0:i1]
        env = np.sin(np.pi * np.clip((tb - max(0.0, t0)) / max(1e-3, min(duration_sec, t1) - max(0.0, t0)), 0.0, 1.0)) ** 1.3
        freqs = chords[b % len(chords)]
        block = np.zeros_like(tb)
        for k, f in enumerate(freqs):
            w = 1.0 / (1.0 + 0.35 * k)
            block += w * np.sin(2.0 * np.pi * f * tb + 0.25 * np.sin(2.0 * np.pi * 0.12 * tb))
        sig[i0:i1] += (block / len(freqs)) * env

    # Master fade-in (2.5s) and fade-out (4.0s)
    fade_in = np.clip(t / 2.5, 0.0, 1.0)
    fade_out = np.clip((duration_sec - t) / 4.0, 0.0, 1.0)
    sig *= fade_in * fade_out
    peak = np.max(np.abs(sig)) + 1e-9
    return (sig / peak) * 0.14  # gentle base level before ducking


def prepare_final_audio(out_wav_path: str, include_music: bool = True) -> str:
    """Clean voice-over, optionally mix subtle ducked ambient pad, and write stereo 44.1kHz WAV."""
    os.makedirs(os.path.dirname(out_wav_path), exist_ok=True)
    clean_voice_wav = out_wav_path + ".voice_clean.wav"

    # High-pass to remove low rumble, mild denoise, gentle compression, loudnorm, pad to TOTAL_DURATION
    cmd = [
        "ffmpeg", "-y", "-i", RAW_AUDIO,
        "-af", (
            "highpass=f=85,"
            "afftdn=nf=-26,"
            "acompressor=threshold=-20dB:ratio=2.5:attack=15:release=180,"
            "loudnorm=I=-16:TP=-1.5:LRA=9,"
            f"apad=whole_dur={TOTAL_DURATION:.3f}"
        ),
        "-ar", "44100", "-ac", "1", "-c:a", "pcm_s16le",
        clean_voice_wav,
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    with wave.open(clean_voice_wav, "rb") as w:
        sr = w.getframerate()
        voice = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768.0

    target_n = int(round(TOTAL_DURATION * sr))
    if len(voice) < target_n:
        voice = np.pad(voice, (0, target_n - len(voice)))
    else:
        voice = voice[:target_n]

    if include_music:
        pad = build_ambient_pad(TOTAL_DURATION, sr=sr)
        # Speech envelope follower for transparent ducking (music dips gently while speaking)
        win = int(sr * 0.12)
        sq = np.convolve(voice ** 2, np.ones(win) / win, mode="same")
        speech_env = np.clip(np.sqrt(sq) / 0.08, 0.0, 1.0)
        duck_gain = 0.055 * (1.0 - 0.62 * speech_env)  # ~ -26dB to -34dB under speech
        mixed = np.clip(voice + pad * duck_gain, -0.97, 0.97)
    else:
        mixed = np.clip(voice, -0.97, 0.97)

    # Export stereo WAV (dual-mono voice + soft ambient)
    pcm = (mixed * 32767.0).astype(np.int16)
    stereo = np.column_stack([pcm, pcm]).ravel()
    with wave.open(out_wav_path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(stereo.tobytes())

    if os.path.exists(clean_voice_wav):
        os.remove(clean_voice_wav)
    return out_wav_path
