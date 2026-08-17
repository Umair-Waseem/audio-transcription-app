# Shared constants, language maps and small helpers used across the package.

from pathlib import Path

import tkinter.font as tkfont

CHUNK_DURATION_SECONDS = 5
SAMPLE_RATE = 16000
PLAYBACK_FRAMES_PER_BUFFER = 1024
DETECTION_CHUNK_COUNT = 3
QUEUE_POLL_MS = 100
THREAD_JOIN_TIMEOUT = 2.0

SUPPORTED_LANGUAGES = {"en": "English", "hi": "Hindi", "ur": "Urdu"}
LANGUAGE_DISPLAY_MAP = {name: code for code, name in SUPPORTED_LANGUAGES.items()}

# NLLB expects language codes that carry the script, unlike Whisper's two-letter codes.
NLLB_LANGUAGE_MAP = {"en": "eng_Latn", "hi": "hin_Deva", "ur": "urd_Arab"}

# Hindi uses a fine-tuned checkpoint because the base model transcribes it poorly.
TRANSCRIPTION_MODEL_MAP = {
    "en": "base",
    "hi": "songzewu/vasista22-whisper-hindi-small-ct2",
    "ur": "base",
}

# Nudges the Hindi model towards Devanagari output instead of romanised text.
HINDI_INITIAL_PROMPT = "हिंदी में बोलें"

# Resolved from the package parent so models/ sits beside app.py, not inside the package.
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
TOKENIZER_PATH = MODELS_DIR / "flores200_sacrebleu_tokenizer_spm.model"
TRANSLATION_MODEL_DIR = MODELS_DIR / "nllb-200-distilled-600M-ct2"

AUDIO_FILE_TYPES = [("Audio Files", "*.wav *.mp3"), ("All Files", "*.*")]


def format_timestamp(start_time: float, end_time: float) -> str:
    return f"{start_time:.1f}s - {end_time:.1f}s"


def downloads_folder() -> Path:
    return Path.home() / "Downloads"


def preferred_output_font() -> tuple:
    # Nastaliq renders Urdu correctly; Arial is a readable fallback for all three languages.
    families = set(tkfont.families())
    return ("Noto Nastaliq Urdu", 12) if "Noto Nastaliq Urdu" in families else ("Arial", 12)
