# Model loading and inference, with each model built once and reused.

import threading

import numpy as np
import sentencepiece as spm
import ctranslate2
from faster_whisper import WhisperModel

from .config import HINDI_INITIAL_PROMPT, NLLB_LANGUAGE_MAP, SUPPORTED_LANGUAGES, TOKENIZER_PATH, TRANSCRIPTION_MODEL_MAP, TRANSLATION_MODEL_DIR

# Whisper and NLLB take seconds to load and hold memory outside Python, so each is built once.
_model_cache = {}
_model_cache_lock = threading.Lock()


def get_cached_model(key: tuple, build):
    with _model_cache_lock:
        if key not in _model_cache:
            _model_cache[key] = build()
        return _model_cache[key]


# --- Section 5: Language detection ---

class LanguageDetector:
    def __init__(self, device: str = "cpu"):
        compute_type = "float32" if device == "cpu" else "float16"
        self.model = get_cached_model(
            ("whisper", "base", device, compute_type),
            lambda: WhisperModel("base", device=device, compute_type=compute_type),
        )

    def detect_language(self, audio_chunks: list) -> str:
        if not audio_chunks:
            raise ValueError("No audio available for language detection.")
        audio = np.concatenate(audio_chunks)
        if audio.size == 0:
            raise ValueError("Audio is empty after concatenation.")
        _, info = self.model.transcribe(audio, language=None, beam_size=5, temperature=0.2)
        if info.language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Detected language '{info.language}' is not supported. "
                f"Supported languages: {', '.join(SUPPORTED_LANGUAGES.values())}."
            )
        return info.language


# --- Section 6: Transcription ---

class Transcriber:
    def __init__(self, device: str = "cpu", language: str = "en"):
        compute_type = "int8" if device == "cpu" else "float16"
        model_name = TRANSCRIPTION_MODEL_MAP.get(language, "base")
        self.model = get_cached_model(
            ("whisper", model_name, device, compute_type),
            lambda: WhisperModel(model_name, device=device, compute_type=compute_type),
        )

    def transcribe_chunk(self, audio_chunk: np.ndarray, language: str) -> str:
        segments, _ = self.model.transcribe(
            audio_chunk,
            language=language,
            vad_filter=True,
            beam_size=1,
            temperature=0.4,
            # Each chunk is a separate call, so this cannot carry context and risks repetition loops.
            condition_on_previous_text=False,
            initial_prompt=HINDI_INITIAL_PROMPT if language == "hi" else None,
        )
        return " ".join(segment.text for segment in segments).strip()


# --- Section 7: Translation ---

class Translator:
    def __init__(self, device: str = "cpu"):
        # Checked up front so a missing model gives a readable message, not a library error.
        missing = [p for p in (TOKENIZER_PATH, TRANSLATION_MODEL_DIR) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                "Translation model files are missing:\n"
                + "\n".join(str(path) for path in missing)
                + "\n\nSee the README section 'Translation Model Setup'."
            )
        self.tokenizer = get_cached_model(("tokenizer", str(TOKENIZER_PATH)), self._build_tokenizer)
        compute_type = "int8" if device == "cpu" else "float16"
        self.translator = get_cached_model(
            ("nllb", device, compute_type),
            lambda: ctranslate2.Translator(
                str(TRANSLATION_MODEL_DIR), device=device, compute_type=compute_type
            ),
        )

    @staticmethod
    def _build_tokenizer():
        tokenizer = spm.SentencePieceProcessor()
        tokenizer.load(str(TOKENIZER_PATH))
        return tokenizer

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        if not text or source_lang == target_lang:
            return text

        source_tag = NLLB_LANGUAGE_MAP[source_lang]
        target_tag = NLLB_LANGUAGE_MAP[target_lang]
        tokens = [source_tag] + self.tokenizer.encode_as_pieces(text) + ["</s>"]
        results = self.translator.translate_batch(
            [tokens], target_prefix=[[target_tag]], beam_size=1, max_decoding_length=256
        )
        # Strips the forced target tag and control tokens the model emits alongside the text.
        hypothesis = [
            token for token in results[0].hypotheses[0]
            if token not in {target_tag, "</s>", "<unk>"}
        ]
        return self.tokenizer.decode_pieces(hypothesis)
