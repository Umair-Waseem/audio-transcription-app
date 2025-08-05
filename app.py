# app.py - Simple Audio Transcription App (Offline, Real-Time)
# Dependencies: Install with pip: faster-whisper transformers pydub pyaudio sentencepiece ctranslate2

import os
import os.path
from pathlib import Path
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import tkinter as tk
import tkinter.font  
from tkinter import filedialog, messagebox, ttk
import numpy as np
import threading
import time
from queue import Queue
import pydub
import pyaudio
from faster_whisper import WhisperModel
import sentencepiece as spm
import ctranslate2
from datetime import datetime

# --- Section 1: Shared Utilities (Constants and Helpers) ---
CHUNK_DURATION_SECONDS = 5        # Fixed chunk size in seconds
CHUNK_OVERLAP_SECONDS = 0         # No overlap to avoid redundant processing
SAMPLE_RATE = 16000               # Audio sample rate for models (Whisper-compatible)
SUPPORTED_LANGUAGES = {'en': 'English', 'hi': 'Hindi', 'ur': 'Urdu'}

LANGUAGE_DISPLAY_MAP = {
    "Urdu": "ur",
    "English": "en",
    "Hindi": "hi"
}

def get_timestamp(start_time: float, end_time: float) -> str:
    """Format timestamp like '0.0s - 5.0s'"""
    return f"{start_time:.1f}s - {end_time:.1f}s"


def handle_error(message: str) -> None:
    """Raise an error with message."""
    raise ValueError(message)


def get_downloads_folder():
    """Get the user's default Downloads folder path"""
    # Try to get the Downloads folder path in a cross-platform way
    if os.name == 'nt':  # Windows
        import ctypes
        from ctypes import windll, wintypes
        CSIDL_PERSONAL = 5       # My Documents
        CSIDL_DOWNLOADS = 0x0011  # Downloads
        
        # First try to get Downloads folder directly
        buf = ctypes.create_unicode_buffer(wintypes.MAX_PATH)
        if windll.shell32.SHGetFolderPathW(None, CSIDL_DOWNLOADS, None, 0, buf):
            # If failed, get My Documents and append Downloads
            if windll.shell32.SHGetFolderPathW(None, CSIDL_PERSONAL, None, 0, buf):
                return os.path.join(os.path.expanduser('~'), 'Downloads')
            return os.path.join(buf.value, 'Downloads')
        return buf.value
    else:  # Mac and Linux
        return os.path.join(os.path.expanduser('~'), 'Downloads')


# --- Section 2: GUI Component ---
class GuiComponent:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Audio Transcription App")
        self.root.geometry("800x500")
        self.root.minsize(600, 400)
        self.root.configure(bg="#f5f5f5")

        style = ttk.Style()
        # Fix layout for background changes
        style.layout("TButton", [
            ('Button.border', {
                'children': [('Button.padding', {
                    'children': [('Button.label', {'sticky': 'nswe'})],
                    'sticky': 'nswe'
                })],
                'sticky': 'nswe'
            })
        ])
        style.theme_use('clam')
        style.configure("TLabel", font=("Helvetica", 10), background="#f5f5f5")
        style.configure("TButton", font=("Helvetica", 10), padding=5)
        style.configure("TRadiobutton", font=("Helvetica", 10))
        style.configure("Active.TButton",
                    background="#ff4d4d",
                    foreground="white",
                    relief="flat")
        style.map("Active.TButton",
                background=[("active", "#ff4d4d"), ("!disabled", "#ff4d4d")],
                foreground=[("active", "white"), ("!disabled", "white")])
        self.control_frame = ttk.Frame(self.root, padding="10 10 10 5")
        self.control_frame.pack(fill=tk.X)

        middle_frame = ttk.Frame(self.root, padding="10 10 10 10")
        middle_frame.pack(fill=tk.BOTH, expand=True)

        self.output_text = tk.Text(
            middle_frame,
            font=("Helvetica", 11),
            wrap=tk.WORD,
            state='disabled',
            bd=0,
            relief="flat",
            bg="#ffffff"
        )
        scrollbar = ttk.Scrollbar(middle_frame, command=self.output_text.yview)
        self.output_text.config(yscrollcommand=scrollbar.set)
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._configure_rtl_text()
        self.display_queue = Queue(maxsize=100)
        self.root.after(100, self._process_display_queue)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _on_close(self):
        self.pipeline.stop_and_reset()
        self.root.destroy()

    def setup_buttons(self):
        load_btn = ttk.Button(
            self.control_frame,
            text="Load File",
            command=self.pipeline.load_and_prepare,
            width=12
        )
        load_btn.pack(side=tk.LEFT, padx=5)

        device_label = ttk.Label(self.control_frame, text="Device:")
        device_label.pack(side=tk.LEFT, padx=(10, 5))

        self.device_var = tk.StringVar(value="cpu")
        cpu_radio = ttk.Radiobutton(
            self.control_frame,
            text="CPU",
            variable=self.device_var,
            value="cpu"
        )
        cpu_radio.pack(side=tk.LEFT, padx=5)
        gpu_radio = ttk.Radiobutton(
            self.control_frame,
            text="GPU",
            variable=self.device_var,
            value="cuda"
        )
        gpu_radio.pack(side=tk.LEFT, padx=5)

        self.play_btn = ttk.Button(
            self.control_frame,
            text="Play",
            command=lambda: self.pipeline.start_processing(self.device_var.get()),
            width=12
        )
        self.play_btn.pack(side=tk.LEFT, padx=10)

        self.pause_btn = ttk.Button(
            self.control_frame,
            text="Pause",
            command=self.pipeline.pause_processing,
            width=12
        )
        self.pause_btn.pack(side=tk.LEFT, padx=5)

        # Add language selection dropdown
        self.output_lang_var = tk.StringVar(value="Urdu")
        lang_label = ttk.Label(self.control_frame, text="Output:")
        lang_label.pack(side=tk.LEFT, padx=(10, 5))
        self.lang_combobox = ttk.Combobox(
            self.control_frame,
            textvariable=self.output_lang_var,
            values=["Urdu", "English", "Hindi"],
            state="readonly",
            width=8
        )
        self.lang_combobox.pack(side=tk.LEFT, padx=5)

        # Add Export button
        export_btn = ttk.Button(
            self.control_frame,
            text="Export",
            command=self.export_transcription,
            width=12
        )
        export_btn.pack(side=tk.LEFT, padx=5)

        self.language_label = ttk.Label(
            self.control_frame,
            text="Detected Language: None"
        )
        self.language_label.pack(side=tk.LEFT, padx=10)
        
        self.file_label = ttk.Label(
            self.control_frame,
            text="Loaded File: None"
        )
        self.file_label.pack(side=tk.LEFT, padx=10)

    def export_transcription(self):
        """Export the transcription text to a file in Downloads folder"""
        try:
            # Get the text from the output widget
            self.output_text.config(state='normal')
            text_content = self.output_text.get("1.0", tk.END)
            self.output_text.config(state='disabled')
            
            if not text_content.strip():
                messagebox.showwarning("No Content", "There is no transcription to export")
                return
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"transcription_{timestamp}.txt"
            
            # Get the Downloads folder path
            downloads_folder = get_downloads_folder()
            
            # Create Downloads folder if it doesn't exist
            os.makedirs(downloads_folder, exist_ok=True)
            
            filepath = os.path.join(downloads_folder, filename)
            
            # Write to file with UTF-8 encoding
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            # Show success message with path to file
            messagebox.showinfo(
                "Export Successful", 
                f"Transcription saved to your Downloads folder:\n{filename}\n\n"
                f"Full path: {filepath}"
            )
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export transcription:\n{str(e)}")

    def update_file_label(self, filename: str):
        name_only = os.path.basename(filename)
        self.file_label.config(text=f"Loaded File: {name_only}")

    def highlight_button(self, active: str):
        if active == "play":
            self.play_btn.config(style="Active.TButton")
            self.pause_btn.config(style="TButton")
            self.lang_combobox.config(state='disabled')
        elif active == "pause":
            self.pause_btn.config(style="Active.TButton")
            self.play_btn.config(style="TButton")
            self.lang_combobox.config(state='readonly')
        else:
            self.play_btn.config(style="TButton")
            self.pause_btn.config(style="TButton")
            self.lang_combobox.config(state='readonly')

    def display_text(self, text: str, timestamp: str):
        self.output_text.config(state='normal')
        self.output_text.insert(tk.END, f"{timestamp}: {text}\n")
        self.output_text.see(tk.END)
        self.output_text.update_idletasks()
        self.output_text.config(state='disabled')

    def clear_output(self):
        self.output_text.config(state='normal')
        self.output_text.delete(1.0, tk.END)
        self.output_text.config(state='disabled')

    def show_error(self, message: str):
        messagebox.showerror("Error", message)

    def _configure_rtl_text(self):
        self.output_text.config(
            font=('Noto Nastaliq Urdu', 12) if 'Noto Nastaliq Urdu' in tkinter.font.families() else ('Arial', 12)
        )
        self.output_text.config(state='disabled')

    def update_language_label(self, language: str):
        self.language_label.config(
            text=f"Detected Language: {SUPPORTED_LANGUAGES.get(language, language)}"
        )

    def _process_display_queue(self):
        while not self.display_queue.empty():
            action, *args = self.display_queue.get()
            if action == 'display':
                self.display_text(*args)
            elif action == 'clear':
                self.clear_output()
            elif action == 'update_lang':
                self.update_language_label(*args)
            elif action == 'error':
                self.show_error(*args)
        self.root.after(100, self._process_display_queue)


# --- Section 3: Audio Playback Handler ---
class AudioHandler:
    def __init__(self):
        self.audio_data = None
        self.chunks = []
        self.current_position = 0
        self.p = pyaudio.PyAudio()
        self.stream = None
        self.play_active = False
        self.play_paused = False
        self.playback_thread = None

    def load_audio_file(self, file_path: str):
        audio = pydub.AudioSegment.from_file(file_path)
        audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
        self.audio_data = np.array(audio.get_array_of_samples(), dtype=np.float32) / 32768.0
        self._create_chunks()
        self.current_position = 0

    def _create_chunks(self):
        chunk_samples = CHUNK_DURATION_SECONDS * SAMPLE_RATE
        step = chunk_samples  # no overlap
        self.chunks = []
        i = 0
        while i < len(self.audio_data):
            end = min(i + chunk_samples, len(self.audio_data))
            chunk = self.audio_data[i:end]
            if chunk.size > 0:
                self.chunks.append(chunk)
            i += step

    def get_initial_chunks_for_detection(self, num_chunks: int = 3):
        size = CHUNK_DURATION_SECONDS * SAMPLE_RATE
        data = self.audio_data[: num_chunks * size]
        chunks = [
            data[i * size: (i + 1) * size]
            for i in range(min(num_chunks, data.shape[0] // size))
        ]
        return chunks or ([data] if data.size > 0 else [])

    def get_chunks_for_transcription(self):
        return self.chunks

    def start_playback(self):
        if self.stream is None:
            self.stream = self.p.open(
                format=pyaudio.paFloat32,
                channels=1,
                rate=SAMPLE_RATE,
                output=True,
                frames_per_buffer=1024
            )
        self.play_active = True
        self.play_paused = False
        if not (self.playback_thread and self.playback_thread.is_alive()):
            self.playback_thread = threading.Thread(target=self._blocking_playback, daemon=True)
            self.playback_thread.start()

    def _blocking_playback(self):
        while self.play_active and self.current_position < len(self.audio_data):
            if self.play_paused:
                time.sleep(0.1)
                continue
            end = min(self.current_position + 1024, len(self.audio_data))
            data = self.audio_data[self.current_position: end]
            self.stream.write(data.tobytes())
            self.current_position = end
        self.play_active = False

    def pause_playback(self):
        self.play_paused = True

    def resume_playback(self, position: int):
        self.current_position = position
        self.play_paused = False
        self.start_playback()

    def stop_and_clear(self):
        self.play_active = False
        self.play_paused = False
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=5)
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        self.stream = None
        self.audio_data = None
        self.chunks = []


# --- Section 4: Language Detection Component ---
class LanguageDetector:
    def __init__(self, device: str = 'cpu'):
        compute_type = "float32" if device=='cpu' else "float16"
        self.model = WhisperModel("tiny", device=device, compute_type=compute_type)

    def detect_language(self, audio_chunks: list[np.ndarray]) -> str:
        if not audio_chunks:
            handle_error("No audio data for detection")
        audio = np.concatenate(audio_chunks)
        if len(audio)==0:
            handle_error("Empty audio after concatenation")
        _, info = self.model.transcribe(
            audio, language=None, beam_size=5, temperature=0.2
        )
        lang = info.language
        if lang not in SUPPORTED_LANGUAGES:
            handle_error(f"Unsupported language detected: {lang}")
        return lang


# --- Section 5: Transcription Component ---
class Transcriber:
    def __init__(self, device: str='cpu', language: str='en'):
        compute_type = "int8" if device=='cpu' else "float16"
        model_name = "base" if language=='en' else "vasista22/whisper-hindi-small"
        self.model = WhisperModel(model_name, device=device, compute_type=compute_type)

    def transcribe_chunk(self, audio_chunk: np.ndarray, language: str) -> str:
        prompt = "" if language=='en' else "हिंदी में बोलें"
        segments, _ = self.model.transcribe(
            audio_chunk,
            language=language,
            vad_filter=True,
            beam_size=1,
            temperature=0.4,
            condition_on_previous_text=True,
            initial_prompt=prompt
        )
        return ' '.join(s.text for s in segments).strip()


# --- Section 6: Translation Component ---
class Translator:
    def __init__(self, device: str='cpu'):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load("models/flores200_sacrebleu_tokenizer_spm.model")
        compute_type = "int8" if device=='cpu' else "float16"
        self.translator = ctranslate2.Translator(
            "models/nllb-200-distilled-600M-ct2",
            device=device,
            compute_type=compute_type
        )

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        if not text:
            return ""
        
        # Map languages to NLLB-200 codes
        lang_map = {
            'en': 'eng_Latn',
            'hi': 'hin_Deva',
            'ur': 'urd_Arab'
        }
        src = lang_map.get(source_lang, 'eng_Latn')
        tgt = lang_map.get(target_lang, 'urd_Arab')
        
        tokens = self.sp.encode_as_pieces(text)
        source = [[src] + tokens + ["</s>"]]
        results = self.translator.translate_batch(
            source,
            target_prefix=[[tgt]],
            beam_size=1,
            max_decoding_length=256
        )
        hyp = results[0].hypotheses[0]
        if tgt in hyp:
            hyp.remove(tgt)
        return self.sp.decode_pieces(hyp)


# --- Section 7: Timestamping and Display Component ---
class TimestampDisplay:
    def __init__(self, gui: GuiComponent):
        self.gui = gui

    def add_and_display(self, text: str, start: float, end: float):
        ts = get_timestamp(start, end)
        self.gui.display_queue.put(('display', text, ts))

    def clear_display(self):
        self.gui.display_queue.put(('clear',))


# --- Section 8: Processing Pipeline (Updated) ---
class ProcessingPipeline:
    def __init__(self, gui: GuiComponent):
        self.gui = gui
        self.audio_handler = AudioHandler()
        self.detected_lang = None
        self.lang_lock = threading.Lock()
        self.lang_condition = threading.Condition(self.lang_lock)
        self.processing_active = False
        self.processing_started_once = False
        self.device = 'cpu'
        self.is_paused = False
        self.current_chunk_i = 0
        self.processing_thread = None
        self.detection_thread = None
        self.pause_lock = threading.Lock()

    def load_and_prepare(self):
        file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
        if file_path:
            self.stop_and_reset()
            self.audio_handler.load_audio_file(file_path)
            self.gui.clear_output()
            self.gui.update_file_label(file_path)

    def start_processing(self, device: str):
        if self.is_paused:
            return self.resume_processing()
        if self.processing_started_once:
            return

        self.device = device
        self.processing_active = True
        self.processing_started_once = True

        self.gui.highlight_button("play")
        self.gui.root.after(100, lambda: self.gui.play_btn.config(state='disabled'))

        self.detection_thread = threading.Thread(target=self._detection_thread, daemon=True)
        self.detection_thread.start()

        self.processing_thread = threading.Thread(target=self._processing_thread, daemon=True)
        self.processing_thread.start()

        self.audio_handler.start_playback()

    def _detection_thread(self):
        try:
            chunks = self.audio_handler.get_initial_chunks_for_detection()
            detector = LanguageDetector(self.device)
            lang = detector.detect_language(chunks)
            with self.lang_lock:
                self.detected_lang = lang
                self.lang_condition.notify_all()
            self.gui.display_queue.put(('update_lang', lang))
        except Exception as e:
            self.gui.display_queue.put(('error', str(e)))
            with self.lang_lock:
                self.lang_condition.notify_all()
            self.stop_and_reset()

    def _processing_thread(self):
        translator = Translator(self.device)
        display = TimestampDisplay(self.gui)
        chunks = self.audio_handler.get_chunks_for_transcription()
        start_time = 0.0

        with self.lang_lock:
            while self.detected_lang is None and self.processing_active:
                self.lang_condition.wait()
            if not self.processing_active:
                self.gui.play_btn.config(state='normal')
                return
            source_lang = self.detected_lang

        transcriber = Transcriber(self.device, source_lang)

        for i in range(self.current_chunk_i, len(chunks)):
            while True:
                with self.pause_lock:
                    if not self.is_paused or not self.processing_active:
                        break
                time.sleep(0.1)
            if not self.processing_active:
                break

            # Get current target language for each chunk
            current_target_lang = LANGUAGE_DISPLAY_MAP[self.gui.output_lang_var.get()]

            text = transcriber.transcribe_chunk(chunks[i], source_lang)
            if not self.processing_active:
                break

            translated_text = translator.translate_text(text, source_lang, current_target_lang)
            end_time = start_time + CHUNK_DURATION_SECONDS
            display.add_and_display(translated_text, start_time, end_time)
            start_time = end_time - CHUNK_OVERLAP_SECONDS
            self.current_chunk_i = i + 1

        self.processing_active = False
        self.gui.play_btn.config(state='normal')

    def pause_processing(self):
        with self.pause_lock:
            self.is_paused = True
        self.gui.highlight_button("pause")
        self.audio_handler.pause_playback()
        self.gui.play_btn.config(state='normal')

    def resume_processing(self):
        if not self.processing_started_once:
            return
        step = (CHUNK_DURATION_SECONDS - CHUNK_OVERLAP_SECONDS) * SAMPLE_RATE
        self.current_chunk_i = int(self.audio_handler.current_position / step)
        with self.pause_lock:
            self.is_paused = False
        self.gui.play_btn.config(state='disabled')
        self.gui.highlight_button("play")
        self.audio_handler.resume_playback(self.audio_handler.current_position)

    def stop_and_reset(self):
        self.processing_active = False
        with self.pause_lock:
            self.is_paused = False
        self.processing_started_once = False
        self.audio_handler.stop_and_clear()
        self.detected_lang = None
        self.current_chunk_i = 0
        self.gui.play_btn.config(state='normal')
        self.gui.highlight_button(None)
        current = threading.current_thread()
        if self.processing_thread and self.processing_thread.is_alive() and self.processing_thread is not current:
            self.processing_thread.join(timeout=5)
        if self.detection_thread and self.detection_thread.is_alive() and self.detection_thread is not current:
            self.detection_thread.join(timeout=5)
        self.processing_thread = None
        self.detection_thread = None


# --- Section 9: Entry Point ---
if __name__ == "__main__":
    gui = GuiComponent()
    pipeline = ProcessingPipeline(gui)
    gui.pipeline = pipeline
    gui.setup_buttons()
    gui.root.mainloop()