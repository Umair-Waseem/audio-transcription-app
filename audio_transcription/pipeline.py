# Coordinates detection, transcription, translation and playback across threads.

import time
import threading
from typing import TYPE_CHECKING

from .audio import AudioHandler
from .config import LANGUAGE_DISPLAY_MAP, SAMPLE_RATE, THREAD_JOIN_TIMEOUT, WIDGET_DISABLED, WIDGET_ENABLED, format_timestamp
from .models import LanguageDetector, Transcriber, Translator

# Imported for type checking only; importing gui at runtime would be circular.
if TYPE_CHECKING:
    from .gui import GuiComponent

class ProcessingPipeline:
    def __init__(self, gui: "GuiComponent"):
        self.gui = gui
        self.audio_handler = AudioHandler(
            on_playback_finished=self._on_playback_finished,
            on_playback_error=self._on_playback_error,
        )

        self.device = "cpu"
        self.target_lang = "ur"
        self.detected_lang = None
        self.current_chunk_index = 0

        self.processing_active = False
        self.run_started = False
        self.is_paused = False
        self.run_id = 0

        self.lang_condition = threading.Condition()
        self.state_lock = threading.Lock()
        self.detection_thread = None
        self.processing_thread = None

    def _is_current_run(self, run_id: int) -> bool:
        return run_id == self.run_id

    def _post(self, action, *args):
        self.gui.display_queue.put((action,) + args)

    def load_file(self, file_path: str):
        # The caller supplies the path, so this layer stays free of any GUI dependency.
        if not file_path:
            return
        try:
            self.stop_and_reset()
            self.audio_handler.load_audio_file(file_path)
        except Exception as exc:
            self.stop_and_reset()
            self.gui.show_error(f"Failed to load audio file:\n{exc}")
            return

        self.gui.clear_pending_updates()
        self.gui.clear_output()
        self.gui.update_file_label(file_path)
        self.gui.update_language_label("None")
        self.gui.set_button_states(WIDGET_ENABLED, WIDGET_DISABLED)
        self.gui.set_restart_state(WIDGET_ENABLED)

    def start_processing(self, device: str):
        if self.is_paused:
            self.resume_processing()
            return
        if self.run_started:
            return
        if not self.audio_handler.has_audio():
            self.gui.show_error("Please load an audio file before pressing Play.")
            return

        self.device = device
        self.target_lang = LANGUAGE_DISPLAY_MAP[self.gui.output_lang_var.get()]
        self.current_chunk_index = 0
        self.audio_handler.current_position = 0
        with self.lang_condition:
            self.detected_lang = None
        # Bumped before activation so any stale worker sees a changed id immediately.
        self.run_id += 1
        run_id = self.run_id
        self.processing_active = True
        self.run_started = True

        self.gui.clear_output()
        self.gui.highlight_button("play")
        self.gui.set_button_states(WIDGET_DISABLED, WIDGET_ENABLED)

        try:
            self.audio_handler.start_playback()
        except Exception as exc:
            self.processing_active = False
            self.run_started = False
            self.gui.highlight_button(None)
            self.gui.set_button_states(WIDGET_ENABLED, WIDGET_DISABLED)
            self.gui.show_error(f"Failed to start playback:\n{exc}")
            return

        self.detection_thread = self._spawn(self._detection_worker, run_id)
        self.processing_thread = self._spawn(self._processing_worker, run_id)

    def _spawn(self, target, run_id: int) -> threading.Thread:
        thread = threading.Thread(target=target, args=(run_id,), daemon=True)
        thread.start()
        return thread

    def restart_processing(self, device: str):
        if not self.audio_handler.has_audio():
            self.gui.show_error("Please load an audio file before restarting.")
            return
        self.stop_and_reset(keep_audio=True)
        self.gui.clear_pending_updates()
        self.gui.clear_output()
        self.gui.update_language_label("None")
        self.gui.highlight_button(None)
        self.gui.set_restart_state(WIDGET_ENABLED)
        self.start_processing(device)

    def _detection_worker(self, run_id: int):
        try:
            detector = LanguageDetector(self.device)
            language = detector.detect_language(self.audio_handler.detection_chunks())
            if not self._is_current_run(run_id):
                return
            with self.lang_condition:
                self.detected_lang = language
                self.lang_condition.notify_all()
            self._post("update_lang", language)
        except Exception as exc:
            if not self._is_current_run(run_id):
                return
            # _reset_after_failure clears the state and wakes the waiting worker.
            self._post("error", str(exc))
            self._reset_after_failure()

    def _wait_for_language(self, run_id: int):
        # Transcription cannot start until detection publishes a language.
        with self.lang_condition:
            while (
                self.detected_lang is None
                and self.processing_active
                and self._is_current_run(run_id)
            ):
                self.lang_condition.wait(timeout=0.5)
            return self.detected_lang

    def _wait_while_paused(self, run_id: int) -> bool:
        # Blocks while paused and reports whether the run is still the active one.
        while True:
            with self.state_lock:
                if not self.is_paused or not self.processing_active:
                    still_running = self.processing_active
                    break
            time.sleep(0.1)
        return still_running and self._is_current_run(run_id)

    def _processing_worker(self, run_id: int):
        try:
            source_lang = self._wait_for_language(run_id)
            if source_lang is None or not self._is_current_run(run_id):
                return

            transcriber = Transcriber(self.device, source_lang)
            translator = None
            chunks = self.audio_handler.chunks
            duration = self.audio_handler.duration_seconds()

            for index in range(self.current_chunk_index, len(chunks)):
                if not self._wait_while_paused(run_id):
                    return

                text = transcriber.transcribe_chunk(chunks[index], source_lang)
                if not self._is_current_run(run_id) or not self.processing_active:
                    return

                if self.target_lang != source_lang:
                    if translator is None:
                        translator = Translator(self.device)
                    text = translator.translate_text(text, source_lang, self.target_lang)
                if not self._is_current_run(run_id) or not self.processing_active:
                    return

                start = self.audio_handler.chunk_start_seconds(index)
                end = min(start + len(chunks[index]) / SAMPLE_RATE, duration)
                self._post("display", text, format_timestamp(start, end))
                self.current_chunk_index = index + 1

            self._finish_run()
        except Exception as exc:
            if not self._is_current_run(run_id):
                return
            self._post("error", str(exc))
            self._reset_after_failure()

    def _finish_run(self):
        # Transcription can outrun playback, so the buttons follow whichever is still going.
        completed = self.processing_active
        self.processing_active = False
        if not completed:
            return
        if self.audio_handler.play_active:
            self._post("button_state", WIDGET_DISABLED, WIDGET_ENABLED)
        else:
            self.run_started = False
            self._post("button_state", WIDGET_ENABLED, WIDGET_DISABLED)
            self._post("highlight", None)

    def pause_processing(self):
        if not self.run_started:
            return
        with self.state_lock:
            self.is_paused = True
        self.audio_handler.pause_playback()
        self.gui.highlight_button("pause")
        self.gui.set_button_states(WIDGET_ENABLED, WIDGET_DISABLED)

    def resume_processing(self):
        if not self.run_started or not self.is_paused:
            return
        with self.state_lock:
            self.is_paused = False
        self.gui.highlight_button("play")
        self.gui.set_button_states(WIDGET_DISABLED, WIDGET_ENABLED)
        try:
            self.audio_handler.resume_playback()
        except Exception as exc:
            self.processing_active = False
            self.run_started = False
            self.gui.highlight_button(None)
            self.gui.set_button_states(
                WIDGET_ENABLED if self.audio_handler.has_audio() else WIDGET_DISABLED, WIDGET_DISABLED
            )
            self.gui.show_error(f"Failed to resume playback:\n{exc}")

    def _on_playback_finished(self):
        if self.processing_active:
            return
        self.run_started = False
        with self.state_lock:
            self.is_paused = False
        self.audio_handler.current_position = 0
        play_state = WIDGET_ENABLED if self.audio_handler.has_audio() else WIDGET_DISABLED
        self._post("button_state", play_state, WIDGET_DISABLED)
        self._post("restart_state", play_state)
        self._post("highlight", None)

    def _on_playback_error(self, exc: Exception):
        # Runs on the playback thread, so the message goes through the queue like any other update.
        self._post("error", f"Audio playback stopped:\n{exc}")
        self._reset_after_failure()

    def _reset_after_failure(self):
        self.run_id += 1
        self.processing_active = False
        self.run_started = False
        self.detected_lang = None
        self.current_chunk_index = 0
        with self.state_lock:
            self.is_paused = False
        with self.lang_condition:
            self.lang_condition.notify_all()
        self.audio_handler.stop_playback()
        self.audio_handler.current_position = 0
        play_state = WIDGET_ENABLED if self.audio_handler.has_audio() else WIDGET_DISABLED
        self._post("button_state", play_state, WIDGET_DISABLED)
        self._post("restart_state", play_state)
        self._post("highlight", None)

    def stop_and_reset(self, keep_audio: bool = False):
        # Bumping run_id first tells any live worker its results are stale.
        self.run_id += 1
        self.processing_active = False
        self.run_started = False
        self.detected_lang = None
        self.current_chunk_index = 0
        with self.state_lock:
            self.is_paused = False
        with self.lang_condition:
            self.lang_condition.notify_all()

        if keep_audio:
            self.audio_handler.stop_playback()
            self.audio_handler.current_position = 0
        else:
            self.audio_handler.stop_and_clear()

        current = threading.current_thread()
        for thread in (self.processing_thread, self.detection_thread):
            if thread and thread.is_alive() and thread is not current:
                thread.join(timeout=THREAD_JOIN_TIMEOUT)
        self.processing_thread = None
        self.detection_thread = None

        if not hasattr(self.gui, "play_btn"):
            return
        self.gui.set_button_states(WIDGET_DISABLED, WIDGET_DISABLED)
        self.gui.highlight_button(None)
        if not keep_audio:
            self.gui.set_restart_state(WIDGET_DISABLED)
            self.gui.update_file_label("None")
            self.gui.update_language_label("None")
