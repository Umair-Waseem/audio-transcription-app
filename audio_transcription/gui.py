# Tkinter interface; worker threads post updates here rather than touching widgets.

import sys
from datetime import datetime
from pathlib import Path
from contextlib import suppress
from queue import Empty, Queue

import tkinter as tk
import tkinter.font as tkfont
from tkinter import filedialog, messagebox, ttk

from .config import AUDIO_FILE_TYPES, LANGUAGE_DISPLAY_MAP, QUEUE_POLL_MS, SUPPORTED_LANGUAGES, downloads_folder


def preferred_output_font() -> tuple:
    # Nastaliq renders Urdu correctly; Arial is a readable fallback for all three languages.
    families = set(tkfont.families())
    return ("Noto Nastaliq Urdu", 12) if "Noto Nastaliq Urdu" in families else ("Arial", 12)

class GuiComponent:
    def __init__(self):
        self.pipeline = None
        self.display_queue = Queue()
        self._queue_job = None

        self.root = tk.Tk()
        self.root.title("Audio Transcription App")
        self.root.geometry("800x500")
        self.root.minsize(600, 400)
        self.root.configure(bg="#f5f5f5")
        self._configure_styles()

        self.control_frame = ttk.Frame(self.root, padding="10 10 10 5")
        self.control_frame.pack(fill=tk.X)

        text_frame = ttk.Frame(self.root, padding="10 10 10 10")
        text_frame.pack(fill=tk.BOTH, expand=True)

        self.output_text = tk.Text(
            text_frame,
            font=preferred_output_font(),
            wrap=tk.WORD,
            state="disabled",
            bd=0,
            relief="flat",
            bg="#ffffff",
        )
        scrollbar = ttk.Scrollbar(text_frame, command=self.output_text.yview)
        self.output_text.config(yscrollcommand=scrollbar.set)
        self.output_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._queue_job = self.root.after(QUEUE_POLL_MS, self._process_display_queue)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _configure_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", font=("Helvetica", 10), background="#f5f5f5")
        style.configure("TButton", font=("Helvetica", 10), padding=5)
        style.configure("TRadiobutton", font=("Helvetica", 10))
        style.configure("Active.TButton", background="#ff4d4d", foreground="white", relief="flat")
        style.map(
            "Active.TButton",
            background=[("active", "#ff4d4d"), ("!disabled", "#ff4d4d")],
            foreground=[("active", "white"), ("!disabled", "white")],
        )

    def build_controls(self):
        ttk.Button(
            self.control_frame, text="Load File", width=12,
            command=self.choose_and_load_file
        ).pack(side=tk.LEFT, padx=5)

        ttk.Label(self.control_frame, text="Device:").pack(side=tk.LEFT, padx=(10, 5))
        self.device_var = tk.StringVar(value="cpu")
        for label, value in (("CPU", "cpu"), ("GPU", "cuda")):
            ttk.Radiobutton(
                self.control_frame, text=label, variable=self.device_var, value=value
            ).pack(side=tk.LEFT, padx=5)

        self.play_btn = ttk.Button(
            self.control_frame, text="Play", width=12, state=tk.DISABLED,
            command=lambda: self.pipeline.start_processing(self.device_var.get())
        )
        self.play_btn.pack(side=tk.LEFT, padx=10)

        self.pause_btn = ttk.Button(
            self.control_frame, text="Pause", width=12, state=tk.DISABLED,
            command=self.pipeline.pause_processing
        )
        self.pause_btn.pack(side=tk.LEFT, padx=5)

        ttk.Label(self.control_frame, text="Output:").pack(side=tk.LEFT, padx=(10, 5))
        self.output_lang_var = tk.StringVar(value="Urdu")
        self.lang_combobox = ttk.Combobox(
            self.control_frame,
            textvariable=self.output_lang_var,
            values=list(LANGUAGE_DISPLAY_MAP.keys()),
            state="readonly",
            width=8,
        )
        self.lang_combobox.pack(side=tk.LEFT, padx=5)

        self.restart_btn = ttk.Button(
            self.control_frame, text="Restart", width=10, state=tk.DISABLED,
            command=lambda: self.pipeline.restart_processing(self.device_var.get())
        )
        self.restart_btn.pack(side=tk.LEFT, padx=5)

        ttk.Button(
            self.control_frame, text="Export", width=12, command=self.export_transcription
        ).pack(side=tk.LEFT, padx=5)

        self.language_label = ttk.Label(self.control_frame, text="Detected Language: None")
        self.language_label.pack(side=tk.LEFT, padx=10)

        self.file_label = ttk.Label(self.control_frame, text="Loaded File: None")
        self.file_label.pack(side=tk.LEFT, padx=10)

    def choose_and_load_file(self):
        # Asking for the path here keeps file dialogs out of the processing layer.
        self.pipeline.load_file(filedialog.askopenfilename(filetypes=AUDIO_FILE_TYPES))

    def export_transcription(self):
        text_content = self.output_text.get("1.0", tk.END).strip()
        if not text_content:
            messagebox.showwarning("No Content", "There is no transcription to export.")
            return

        # Offers a save dialogue so the user chooses the destination, defaulting to Downloads.
        suggested = f"transcription_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        file_path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            initialfile=suggested,
            initialdir=str(downloads_folder()),
            filetypes=[("Text Files", "*.txt")],
        )
        if not file_path:
            return

        try:
            Path(file_path).write_text(text_content + "\n", encoding="utf-8")
        except OSError as exc:
            messagebox.showerror("Export Error", f"Failed to export transcription:\n{exc}")
            return
        messagebox.showinfo("Export Successful", f"Transcription saved to:\n{file_path}")

    def update_file_label(self, filename: str):
        name = Path(filename).name if filename != "None" else "None"
        self.file_label.config(text=f"Loaded File: {name}")

    def update_language_label(self, language: str):
        self.language_label.config(
            text=f"Detected Language: {SUPPORTED_LANGUAGES.get(language, language)}"
        )

    def highlight_button(self, active):
        # Language choice is locked while a run is active because changing it mid-run has no effect.
        self.play_btn.config(style="Active.TButton" if active == "play" else "TButton")
        self.pause_btn.config(style="Active.TButton" if active == "pause" else "TButton")
        self.lang_combobox.config(state="disabled" if active else "readonly")

    def set_button_states(self, play_state, pause_state):
        self.play_btn.config(state=play_state)
        self.pause_btn.config(state=pause_state)

    def set_restart_state(self, state):
        self.restart_btn.config(state=state)

    def display_text(self, text: str, timestamp: str):
        self.output_text.config(state="normal")
        self.output_text.insert(tk.END, f"{timestamp}: {text}\n")
        self.output_text.see(tk.END)
        self.output_text.config(state="disabled")

    def clear_output(self):
        self.output_text.config(state="normal")
        self.output_text.delete("1.0", tk.END)
        self.output_text.config(state="disabled")

    def clear_pending_updates(self):
        while True:
            try:
                self.display_queue.get_nowait()
            except Empty:
                return

    def show_error(self, message: str):
        messagebox.showerror("Error", message)

    def _process_display_queue(self):
        # Worker threads never touch widgets directly; they post actions here instead.
        handlers = {
            "display": self.display_text,
            "update_lang": self.update_language_label,
            "error": self.show_error,
            "highlight": self.highlight_button,
            "button_state": self.set_button_states,
            "restart_state": self.set_restart_state,
        }
        try:
            while True:
                try:
                    action, *args = self.display_queue.get_nowait()
                except Empty:
                    break
                handler = handlers.get(action)
                if handler:
                    handler(*args)
        except Exception as exc:
            # A failing handler is reported and skipped rather than killing the update loop.
            print(f"Display update failed: {exc}", file=sys.stderr)
        finally:
            # Rescheduled in finally so one bad update cannot stop every future one.
            self._queue_job = self.root.after(QUEUE_POLL_MS, self._process_display_queue)

    def on_close(self):
        # Cancels the pending poll first, otherwise it fires against a destroyed interpreter.
        if self._queue_job is not None:
            with suppress(tk.TclError):
                self.root.after_cancel(self._queue_job)
            self._queue_job = None
        if self.pipeline:
            self.pipeline.stop_and_reset()
        self.root.destroy()
