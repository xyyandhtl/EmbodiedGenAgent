"""
vlm_robot_agent/io/speech_io.py
===============================
• TTS  : Coqui-TTS VITS Spanish (fast, offline)
• STT  : Whisper-tiny (offline, no FP16 warnings)
"""

from __future__ import annotations
from pathlib import Path
import tempfile, os, warnings

import sounddevice as sd
import soundfile as sf
import numpy as np

# ------------------ QUICK CONFIG ------------------
GPU: bool = False          # True if you have a GPU
TTS_SPEED: float = 1.3     # 1.0 = normal, >1 = faster
# ---------------------------------------------------

# -------------------  T  T  S  ---------------------
from TTS.api import TTS
_TTS: TTS | None = None


def _get_tts() -> TTS:
    global _TTS
    if _TTS is None:
        _TTS = TTS("tts_models/es/css10/vits", gpu=GPU)
    return _TTS


def _clean(text: str) -> str:
    return text.replace("¿", "").replace("¡", "")


def speak(text: str, *, block: bool = True) -> None:
    tts = _get_tts()
    wav = tts.tts(_clean(text), speed=TTS_SPEED)   # ← accelerate synthesis
    sd.play(wav, tts.synthesizer.output_sample_rate)
    if block:
        sd.wait()


def tts_to_file(text: str, path: str | Path) -> Path:
    tts = _get_tts()
    wav = tts.tts(_clean(text), speed=TTS_SPEED)
    path = Path(path)
    sf.write(str(path), wav, tts.synthesizer.output_sample_rate)
    return path


# -------------------  S  T  T  ---------------------
import whisper
import torch
warnings.filterwarnings(
    "ignore",
    message="FP16 is not supported on CPU; using FP32 instead",
    category=UserWarning,
)

  
device = "cuda" if torch.cuda.is_available() else "cpu"
_STT = whisper.load_model("tiny", device=device) # ~60 MB CPU

def transcribe_file(path: str | Path) -> str:
    return _STT.transcribe(str(path), language="es", fp16=False)["text"].strip()


def listen(
    seconds: int = 5,
    *,
    samplerate: int = 16_000,
    channels: int = 1,
) -> str | None:
    audio = sd.rec(int(seconds * samplerate), samplerate, channels, dtype="float32")
    sd.wait()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        sf.write(tmp.name, audio, samplerate)
        text = transcribe_file(tmp.name)
    os.unlink(tmp.name)
    return text or None


# -------------------  D E M O  ---------------------
if __name__ == "__main__":
    print("▶ TTS Test (fast voice)…")
    speak("Hello, I am the new Coqui voice. Do you hear the ñ correctly?")

    print("▶ STT Test (4 s)…")
    text = listen(4)
    print("➡ Recognized text:", text)
