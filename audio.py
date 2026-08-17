# Audio decoding, chunking and playback on a background thread.

import time
import threading
from contextlib import suppress

import numpy as np
import pydub
import pyaudio

from .config import CHUNK_DURATION_SECONDS, DETECTION_CHUNK_COUNT, PLAYBACK_FRAMES_PER_BUFFER, SAMPLE_RATE, THREAD_JOIN_TIMEOUT

class AudioHandler:
    def __init__(self, on_playback_finished=None, on_playback_error=None):
        self.audio_data = None
        self.chunks = []
        self.current_position = 0
        self.audio_interface = None
        self.stream = None
        self.play_active = False
        self.play_paused = False
        self.playback_thread = None
        self.stop_requested = False
        self.on_playback_finished = on_playback_finished
        self.on_playback_error = on_playback_error

    def load_audio_file(self, file_path: str):
        audio = pydub.AudioSegment.from_file(file_path)
        audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2)
        samples = np.array(audio.get_array_of_samples(), dtype=np.float32)
        # A file can decode cleanly yet hold no audio, so it is rejected here rather than at playback.
        if samples.size == 0:
            raise ValueError("The file contains no audio data.")
        # Scales signed 16-bit samples into the -1.0 to 1.0 range the models and PyAudio expect.
        self.audio_data = samples / float(1 << 15)
        self._create_chunks()
        self.current_position = 0

    def _create_chunks(self):
        chunk_samples = CHUNK_DURATION_SECONDS * SAMPLE_RATE
        self.chunks = [
            self.audio_data[start:start + chunk_samples]
            for start in range(0, len(self.audio_data), chunk_samples)
        ]

    def chunk_start_seconds(self, index: int) -> float:
        # Derived from the chunk index so timestamps always match how chunks were cut.
        return index * CHUNK_DURATION_SECONDS

    def detection_chunks(self) -> list:
        return self.chunks[:DETECTION_CHUNK_COUNT]

    def has_audio(self) -> bool:
        return self.audio_data is not None and self.audio_data.size > 0

    def duration_seconds(self) -> float:
        return len(self.audio_data) / SAMPLE_RATE if self.has_audio() else 0.0

    def start_playback(self):
        if not self.has_audio():
            raise ValueError("Load an audio file before starting playback.")
        try:
            if self.audio_interface is None:
                self.audio_interface = pyaudio.PyAudio()
            if self.stream is None:
                self.stream = self.audio_interface.open(
                    format=pyaudio.paFloat32,
                    channels=1,
                    rate=SAMPLE_RATE,
                    output=True,
                    frames_per_buffer=PLAYBACK_FRAMES_PER_BUFFER,
                )
        except Exception:
            self.stop_playback()
            raise

        self.stop_requested = False
        self.play_active = True
        self.play_paused = False
        if not (self.playback_thread and self.playback_thread.is_alive()):
            self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
            self.playback_thread.start()

    def _playback_loop(self):
        reached_end = False
        failure = None
        try:
            while self.play_active and self.current_position < len(self.audio_data):
                if self.play_paused:
                    time.sleep(0.1)
                    continue
                end = min(self.current_position + PLAYBACK_FRAMES_PER_BUFFER, len(self.audio_data))
                self.stream.write(self.audio_data[self.current_position:end].tobytes())
                self.current_position = end
            reached_end = not self.stop_requested and self.current_position >= len(self.audio_data)
        except Exception as exc:
            # A dead audio device must surface as a message, not as silence with a live UI.
            failure = exc
        finally:
            self.play_active = False
            if failure is not None:
                self._release_stream()
                if self.on_playback_error:
                    self.on_playback_error(failure)
            elif reached_end:
                self._release_stream()
                if self.on_playback_finished:
                    self.on_playback_finished()

    def pause_playback(self):
        self.play_paused = True

    def resume_playback(self):
        self.play_paused = False
        self.start_playback()

    def _release_stream(self):
        # Separated from stop_playback so the playback thread can close its own stream safely.
        stream, self.stream = self.stream, None
        if stream:
            with suppress(Exception):
                if stream.is_active():
                    stream.stop_stream()
            with suppress(Exception):
                stream.close()
        interface, self.audio_interface = self.audio_interface, None
        if interface:
            with suppress(Exception):
                interface.terminate()

    def stop_playback(self):
        self.stop_requested = True
        self.play_active = False
        self.play_paused = False
        thread = self.playback_thread
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=THREAD_JOIN_TIMEOUT)
        self.playback_thread = None
        self._release_stream()

    def stop_and_clear(self):
        self.stop_playback()
        self.audio_data = None
        self.chunks = []
        self.current_position = 0
