import numpy as np
import soundfile as sf
import librosa
from pathlib import Path

def load_audio(path: str | Path, target_sr: int = 24000) -> np.ndarray:
    """Loads an audio file, mixes to mono, resamples, and normalizes."""
    y, sr = sf.read(path, always_2d=True)
    y = y.mean(axis=1)  # Mix to mono
    
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        
    max_val = np.max(np.abs(y))
    if max_val > 1.0:
        y = y / max_val
        
    return y.astype(np.float32)

def float_to_s16le(audio: np.ndarray) -> np.ndarray:
    """Converts float32 audio to int16 (s16le) PCM format."""
    return np.clip(audio * 32767.0, -32768, 32767).astype(np.int16)