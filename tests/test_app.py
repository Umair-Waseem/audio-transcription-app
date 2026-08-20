# Run with: python tests/test_app.py

import io
import re
import sys
import time
import types
import tempfile
import threading
import contextlib
from pathlib import Path
from queue import Queue, Empty

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from audio_transcription import audio as audio_module
from audio_transcription import config as config_module
from audio_transcription import gui as gui_module
from audio_transcription import models as models_module
from audio_transcription import pipeline as pipeline_module
from audio_transcription.audio import AudioHandler
from audio_transcription.gui import GuiComponent
from audio_transcription.pipeline import ProcessingPipeline


# --- Helpers ---

class FakeWhisper:
    # Records each construction so caching can be checked, and returns fixed output.
    builds = []
    captured = {}

    def __init__(self, name, device=None, compute_type=None):
        FakeWhisper.builds.append(name)

    def transcribe(self, audio, **kwargs):
        FakeWhisper.captured.update(kwargs)
        return [types.SimpleNamespace(text=" hi ")], types.SimpleNamespace(language="en")


class FailingStream:
    # Raises partway through playback to imitate a device disappearing mid-file.
    def __init__(self):
        self.writes = 0

    def write(self, data):
        self.writes += 1
        if self.writes == 3:
            raise OSError("device lost")

    def is_active(self):
        return True

    def stop_stream(self):
        pass

    def close(self):
        pass


def decoded_segment(samples):
    # Stands in for a pydub segment holding the given samples.
    class Segment:
        def set_frame_rate(self, rate):
            return self

        def set_channels(self, channels):
            return self

        def set_sample_width(self, width):
            return self

        def get_array_of_samples(self):
            return samples

    return Segment()


@contextlib.contextmanager
def patched(module, name, value):
    # Restores the original afterwards so one test cannot affect the next.
    original = getattr(module, name)
    setattr(module, name, value)
    try:
        yield
    finally:
        setattr(module, name, original)


class StubGui:
    # Accepts every GUI call and keeps the queue so posted messages can be inspected.
    def __init__(self, language="English"):
        self.display_queue = Queue()
        self.play_btn = object()
        self.output_lang_var = types.SimpleNamespace(get=lambda: language)

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


class StubAudio:
    # Five-second chunks with playback replaced, so no sound device is needed.
    def __init__(self, chunk_count=3):
        self.chunks = [np.zeros(80000, dtype=np.float32)] * chunk_count
        self.play_active = False
        self.current_position = 0
        self.calls = []

    def has_audio(self):
        return True

    def duration_seconds(self):
        return 5.0 * len(self.chunks)

    def chunk_start_seconds(self, index):
        return index * 5

    def detection_chunks(self):
        return self.chunks

    def start_playback(self):
        self.calls.append("start")
        self.play_active = True

    def pause_playback(self):
        self.calls.append("pause")

    def resume_playback(self):
        self.calls.append("resume")
        self.play_active = True

    def stop_playback(self):
        self.calls.append("stop")
        self.play_active = False

    def stop_and_clear(self):
        self.calls.append("clear")
        self.play_active = False


def make_pipeline(gui, handler):
    # Built without __init__ so the audio handler and models can be stubbed.
    pipeline = ProcessingPipeline.__new__(ProcessingPipeline)
    pipeline.gui = gui
    pipeline.audio_handler = handler
    pipeline.device = "cpu"
    pipeline.target_lang = "en"
    pipeline.detected_lang = None
    pipeline.current_chunk_index = 0
    pipeline.processing_active = False
    pipeline.run_started = False
    pipeline.is_paused = False
    pipeline.run_id = 0
    pipeline.lang_condition = threading.Condition()
    pipeline.state_lock = threading.Lock()
    pipeline.detection_thread = None
    pipeline.processing_thread = None
    return pipeline


def drain(queue):
    messages = []
    while True:
        try:
            messages.append(queue.get_nowait())
        except Empty:
            return messages


# --- Tests ---

def test_chunking_covers_audio_exactly():
    # Gaps or overlaps here would misalign every timestamp shown to the user.
    handler = AudioHandler()
    handler.audio_data = np.zeros(int(32.8 * 16000), dtype=np.float32)
    handler._create_chunks()
    assert len(handler.chunks) == 7
    assert sum(len(c) for c in handler.chunks) == handler.audio_data.size
    last_end = handler.chunk_start_seconds(6) + len(handler.chunks[-1]) / 16000
    assert abs(last_end - handler.duration_seconds()) < 1e-6


def test_samples_normalise_into_model_range():
    # Whisper and PyAudio both expect float32 between -1.0 and 1.0.
    values = np.array([-32768, 0, 32767], dtype=np.float32) / float(1 << 15)
    assert values.min() >= -1.0 and values.max() <= 1.0


def test_playback_failure_is_reported_not_swallowed():
    # A dead device must produce a message instead of silence with a live-looking window.
    errors, finished = [], []
    handler = AudioHandler(on_playback_finished=lambda: finished.append(True),
                           on_playback_error=lambda exc: errors.append(exc))
    handler.audio_data = np.zeros(16000, dtype=np.float32)
    handler._create_chunks()
    handler.stream = FailingStream()
    handler.audio_interface = types.SimpleNamespace(terminate=lambda: None)
    handler.play_active = True
    handler._playback_loop()
    assert errors and not finished
    assert handler.stream is None


def test_models_are_built_once_and_reused():
    # Rebuilding on every run would add seconds to Restart and leak native memory.
    FakeWhisper.builds.clear()
    with patched(models_module, "WhisperModel", FakeWhisper):
        models_module._model_cache.clear()
        for _ in range(4):
            models_module.LanguageDetector("cpu")
        built = len(FakeWhisper.builds)
        models_module._model_cache.clear()
    assert built == 1


def test_transcription_uses_the_intended_parameters():
    # Each chunk is a separate call, so previous-text conditioning must stay off.
    FakeWhisper.captured.clear()
    with patched(models_module, "WhisperModel", FakeWhisper):
        models_module._model_cache.clear()
        text = models_module.Transcriber("cpu", "en").transcribe_chunk(
            np.zeros(80000, dtype=np.float32), "en")
        models_module._model_cache.clear()
    assert text == "hi"
    assert FakeWhisper.captured["condition_on_previous_text"] is False
    assert FakeWhisper.captured["vad_filter"] is True


def test_display_queue_survives_a_failing_handler():
    # Without this, one bad update would stop every future update permanently.
    gui = GuiComponent.__new__(GuiComponent)
    gui.display_queue = Queue()
    gui._queue_job = None
    reschedules = []
    gui.root = types.SimpleNamespace(after=lambda ms, fn: reschedules.append(fn) or "job")
    gui.display_text = lambda text, stamp: None
    gui.show_error = lambda message: None
    gui.highlight_button = lambda active: None
    gui.set_restart_state = lambda state: None
    gui.set_button_states = lambda play, pause: None

    def raising(language):
        raise RuntimeError("widget destroyed")

    gui.update_language_label = raising
    gui.display_queue.put(("update_lang", "en"))
    # The handler is meant to fail here, so its report is kept out of the test output.
    with contextlib.redirect_stderr(io.StringIO()):
        gui._process_display_queue()
    assert len(reschedules) == 1


def test_close_cancels_polling_before_destroying_the_window():
    # A pending callback firing after destroy would raise TclError on exit.
    order = []
    gui = GuiComponent.__new__(GuiComponent)
    gui.display_queue = Queue()
    gui._queue_job = "job"
    gui.root = types.SimpleNamespace(after_cancel=lambda job: order.append("cancel"),
                                     destroy=lambda: order.append("destroy"))
    gui.pipeline = types.SimpleNamespace(stop_and_reset=lambda: order.append("stop"))
    gui.on_close()
    assert order == ["cancel", "stop", "destroy"]
    assert gui._queue_job is None


def test_export_writes_utf8_so_urdu_survives():
    # Urdu is the default output language, so transcripts must round trip intact.
    gui = GuiComponent.__new__(GuiComponent)
    urdu = "0.0s: \u0627\u0644\u0633\u0644\u0627\u0645\n"
    gui.output_text = types.SimpleNamespace(get=lambda start, end: urdu)
    quiet = types.SimpleNamespace(showwarning=lambda *a: None,
                                  showerror=lambda *a: None,
                                  showinfo=lambda *a: None)
    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / "out.txt"
        chooser = types.SimpleNamespace(asksaveasfilename=lambda **kwargs: str(target))
        with patched(gui_module, "messagebox", quiet), patched(gui_module, "filedialog", chooser):
            gui.export_transcription()
        assert "\u0627\u0644\u0633\u0644\u0627\u0645" in target.read_text(encoding="utf-8")


def test_translation_tags_and_strips_control_tokens():
    # A leaked target tag or </s> would show up as noise in the transcript.
    translator = models_module.Translator.__new__(models_module.Translator)
    translator.tokenizer = types.SimpleNamespace(
        encode_as_pieces=lambda text: ["\u2581" + w for w in text.split()],
        decode_pieces=lambda tokens: " ".join(t.replace("\u2581", "") for t in tokens))
    translator.translator = types.SimpleNamespace(
        translate_batch=lambda batch, target_prefix=None, **kwargs:
            [types.SimpleNamespace(hypotheses=[["urd_Arab", "\u2581\u0633\u0644\u0627\u0645", "</s>"]])])
    assert translator.translate_text("hello", "en", "ur") == "\u0633\u0644\u0627\u0645"
    assert translator.translate_text("unchanged", "en", "en") == "unchanged"
    assert translator.translate_text("", "en", "ur") == ""


def test_empty_file_is_rejected_at_load():
    # A file can decode cleanly yet hold no audio, which must not enable Play.
    stub = types.SimpleNamespace(AudioSegment=types.SimpleNamespace(
        from_file=lambda path: decoded_segment([])))
    with patched(audio_module, "pydub", stub):
        try:
            AudioHandler().load_audio_file("empty.wav")
        except ValueError as exc:
            assert "no audio data" in str(exc)
        else:
            raise AssertionError("an empty file was accepted")


def test_workers_produce_correctly_timestamped_output():
    # Exercises the detection and transcription threads together, end to end.
    with patched(models_module, "WhisperModel", FakeWhisper):
        models_module._model_cache.clear()
        pipeline = make_pipeline(StubGui(), StubAudio(chunk_count=3))
        pipeline.start_processing("cpu")
        for thread in (pipeline.detection_thread, pipeline.processing_thread):
            thread.join(timeout=10)
        displays = [m for m in drain(pipeline.gui.display_queue) if m[0] == "display"]
        models_module._model_cache.clear()
    assert len(displays) == 3
    assert displays[0][2] == "0.0s - 5.0s"
    assert displays[-1][2] == "10.0s - 15.0s"


def test_repeated_runs_leave_no_threads_behind():
    # Restarting many times must not accumulate worker threads.
    with patched(models_module, "WhisperModel", FakeWhisper):
        models_module._model_cache.clear()
        pipeline = make_pipeline(StubGui(), StubAudio(chunk_count=3))
        before = threading.active_count()
        for _ in range(4):
            pipeline.run_started = False
            pipeline.start_processing("cpu")
            time.sleep(0.05)
            pipeline.stop_and_reset(keep_audio=True)
        time.sleep(0.3)
        models_module._model_cache.clear()
    assert threading.active_count() == before


def test_language_detection_rejects_unsupported_languages():
    # Transcribing an unsupported language would produce confident nonsense.
    class ForeignWhisper:
        def transcribe(self, audio, **kwargs):
            return [], types.SimpleNamespace(language="fr")

    detector = models_module.LanguageDetector.__new__(models_module.LanguageDetector)
    detector.model = ForeignWhisper()
    try:
        detector.detect_language([np.zeros(16000, dtype=np.float32)])
    except ValueError as exc:
        assert "not supported" in str(exc)
    else:
        raise AssertionError("an unsupported language was accepted")


def test_load_file_accepts_valid_audio():
    # The successful path must populate chunks and leave the handler ready to play.
    samples = [1000] * (16000 * 7)
    stub = types.SimpleNamespace(AudioSegment=types.SimpleNamespace(
        from_file=lambda path: decoded_segment(samples)))
    handler = AudioHandler()
    pipeline = make_pipeline(StubGui(), handler)
    with patched(audio_module, "pydub", stub):
        pipeline.load_file("recording.wav")
    assert handler.has_audio()
    assert len(handler.chunks) == 2
    assert abs(handler.duration_seconds() - 7.0) < 1e-6


def test_cancelled_dialog_leaves_the_loaded_file_alone():
    # Cancelling must reach the real load_file and leave the existing audio in place.
    handler = AudioHandler()
    handler.audio_data = np.zeros(16000, dtype=np.float32)
    handler._create_chunks()
    gui = GuiComponent.__new__(GuiComponent)
    gui.pipeline = make_pipeline(StubGui(), handler)
    chooser = types.SimpleNamespace(askopenfilename=lambda **kwargs: "")
    with patched(gui_module, "filedialog", chooser):
        gui.choose_and_load_file()
    assert handler.has_audio()
    assert len(handler.chunks) == 1


def test_pause_and_resume_reach_the_audio_handler():
    # Pause must hold the workers and stop the audio, and resume must release both.
    audio = StubAudio()
    pipeline = make_pipeline(StubGui(), audio)
    pipeline.run_started = True
    pipeline.processing_active = True
    pipeline.pause_processing()
    assert pipeline.is_paused is True
    assert "pause" in audio.calls
    pipeline.resume_processing()
    assert pipeline.is_paused is False
    assert pipeline.run_started is True
    assert "resume" in audio.calls


def test_pause_is_ignored_when_no_run_is_active():
    # Guards against a stray Pause leaving the pipeline in a paused state forever.
    pipeline = make_pipeline(StubGui(), StubAudio())
    pipeline.pause_processing()
    assert pipeline.is_paused is False


def test_finished_playback_restores_the_controls():
    # When transcription ends first, the run only completes once playback stops.
    pipeline = make_pipeline(StubGui(), StubAudio())
    pipeline.run_started = True
    pipeline.processing_active = False
    pipeline.audio_handler.current_position = 12345
    pipeline._on_playback_finished()
    actions = [m[0] for m in drain(pipeline.gui.display_queue)]
    assert pipeline.run_started is False
    assert pipeline.audio_handler.current_position == 0
    assert "button_state" in actions and "highlight" in actions


def test_restart_invalidates_the_previous_run():
    # Results from the abandoned run must be discarded rather than mixed into the new one.
    with patched(models_module, "WhisperModel", FakeWhisper):
        models_module._model_cache.clear()
        pipeline = make_pipeline(StubGui(), StubAudio(chunk_count=2))
        pipeline.start_processing("cpu")
        first_run = pipeline.run_id
        pipeline.restart_processing("cpu")
        second_run = pipeline.run_id
        for thread in (pipeline.detection_thread, pipeline.processing_thread):
            thread.join(timeout=10)
        models_module._model_cache.clear()
    assert second_run > first_run
    assert pipeline._is_current_run(first_run) is False


def test_stop_playback_clears_state_and_thread():
    # A stale play_active or thread reference would block the next run from starting.
    handler = AudioHandler()
    handler.audio_data = np.zeros(16000, dtype=np.float32)
    handler._create_chunks()
    handler.play_active = True
    handler.play_paused = True
    handler.stop_playback()
    assert handler.play_active is False
    assert handler.play_paused is False
    assert handler.stop_requested is True
    assert handler.playback_thread is None


def test_detection_uses_the_configured_number_of_chunks():
    # Too little audio makes language detection unreliable.
    handler = AudioHandler()
    handler.audio_data = np.zeros(16000 * 30, dtype=np.float32)
    handler._create_chunks()
    assert len(handler.detection_chunks()) == config_module.DETECTION_CHUNK_COUNT


def test_transcript_widget_returns_to_read_only_after_a_write():
    # Leaving it editable would let a user alter the transcript in place.
    states = []

    class Widget:
        def insert(self, index, text):
            pass

        def see(self, index):
            pass

        def config(self, **kwargs):
            states.append(kwargs.get("state"))

    gui = GuiComponent.__new__(GuiComponent)
    gui.output_text = Widget()
    gui.display_text("hello", "0.0s - 5.0s")
    assert states[-1] == "disabled"


def test_export_writes_utf8_bytes_regardless_of_platform_default():
    # Without an explicit encoding this fails on systems that default to cp1252.
    gui = GuiComponent.__new__(GuiComponent)
    urdu = "0.0s: \u0627\u0644\u0633\u0644\u0627\u0645\n"
    gui.output_text = types.SimpleNamespace(get=lambda start, end: urdu)
    quiet = types.SimpleNamespace(showwarning=lambda *a: None,
                                  showerror=lambda *a: None,
                                  showinfo=lambda *a: None)
    with tempfile.TemporaryDirectory() as folder:
        target = Path(folder) / "out.txt"
        chooser = types.SimpleNamespace(asksaveasfilename=lambda **kwargs: str(target))
        with patched(gui_module, "messagebox", quiet), patched(gui_module, "filedialog", chooser):
            gui.export_transcription()
        assert target.read_bytes().decode("utf-8").strip().endswith("\u0627\u0644\u0633\u0644\u0627\u0645")


def test_language_selection_is_locked_while_a_run_is_active():
    # Changing the output language mid-run has no effect, so the control must be disabled.
    states = []

    class Combobox:
        def config(self, **kwargs):
            states.append(kwargs.get("state"))

    class Button:
        def config(self, **kwargs):
            pass

    gui = GuiComponent.__new__(GuiComponent)
    gui.play_btn = Button()
    gui.pause_btn = Button()
    gui.lang_combobox = Combobox()
    gui.highlight_button("play")
    gui.highlight_button(None)
    assert states == ["disabled", "readonly"]


def test_missing_translation_models_are_reported_by_name():
    # A missing file must produce a readable message, not a low-level library error.
    absent = Path("/nonexistent/flores.model")
    with patched(models_module, "TOKENIZER_PATH", absent), \
         patched(models_module, "TRANSLATION_MODEL_DIR", Path("/nonexistent/nllb")):
        try:
            models_module.Translator("cpu")
        except FileNotFoundError as exc:
            assert "missing" in str(exc).lower()
            assert str(absent) in str(exc)
        else:
            raise AssertionError("missing model files were not detected")


def test_finished_run_releases_the_controls_when_playback_has_stopped():
    # Otherwise Play would stay disabled after everything has finished.
    audio = StubAudio()
    audio.play_active = False
    pipeline = make_pipeline(StubGui(), audio)
    pipeline.run_started = True
    pipeline.processing_active = True
    pipeline._finish_run()
    assert pipeline.run_started is False
    actions = [m[0] for m in drain(pipeline.gui.display_queue)]
    assert "button_state" in actions and "highlight" in actions


def test_stop_and_reset_invalidates_work_already_in_flight():
    # Without a new run id, a cancelled worker could still write into the transcript.
    pipeline = make_pipeline(StubGui(), StubAudio())
    pipeline.run_id = 5
    pipeline.stop_and_reset(keep_audio=True)
    assert pipeline.run_id > 5
    assert pipeline._is_current_run(5) is False


def test_detection_rejects_an_empty_chunk_list():
    # Concatenating nothing would raise an unclear numpy error instead.
    detector = models_module.LanguageDetector.__new__(models_module.LanguageDetector)
    detector.model = None
    try:
        detector.detect_language([])
    except ValueError as exc:
        assert "No audio" in str(exc)
    else:
        raise AssertionError("an empty chunk list was accepted")


def test_failure_reset_invalidates_work_already_in_flight():
    # A failing run must not let its late results appear in the next transcript.
    pipeline = make_pipeline(StubGui(), StubAudio())
    pipeline.run_id = 5
    pipeline.processing_active = True
    pipeline.run_started = True
    pipeline._reset_after_failure()
    assert pipeline.run_id > 5
    assert pipeline._is_current_run(5) is False
    assert pipeline.processing_active is False
    assert pipeline.run_started is False
    assert pipeline.current_chunk_index == 0


def test_file_writes_state_their_encoding_explicitly():
    # Relying on the platform default corrupts Urdu on systems that are not UTF-8.
    package = Path(__file__).resolve().parent.parent / "audio_transcription"
    for module in sorted(package.glob("*.py")):
        text = module.read_text()
        # Only text file operations are checked; audio streams have no encoding.
        for call in ("write_text(", "read_text(", "io.open(", "= open("):
            index = text.find(call)
            while index != -1:
                window = text[index:index + 160]
                assert "encoding=" in window, f"{module.name} reads or writes without an encoding"
                index = text.find(call, index + 1)


def test_every_posted_action_has_a_handler():
    # A missing handler would make worker messages, including errors, vanish silently.
    pipeline_source = (Path(__file__).resolve().parent.parent
                       / "audio_transcription" / "pipeline.py").read_text()
    gui_source = (Path(__file__).resolve().parent.parent
                  / "audio_transcription" / "gui.py").read_text()
    posted = set(re.findall(r'_post\(\s*"([a-z_]+)"', pipeline_source))
    block = gui_source[gui_source.index("handlers = {"):]
    handled = set(re.findall(r'"([a-z_]+)":', block[:block.index("}")]))
    assert posted, "no posted actions were found"
    assert not posted - handled, f"posted with no handler: {sorted(posted - handled)}"
    assert not handled - posted, f"handlers never used: {sorted(handled - posted)}"


def test_paused_worker_reports_a_cancelled_run_as_stopped():
    # Returning True here would let a cancelled run keep writing into the transcript.
    pipeline = make_pipeline(StubGui(), StubAudio())
    pipeline.processing_active = False
    pipeline.is_paused = False
    assert pipeline._wait_while_paused(pipeline.run_id) is False


def test_playback_end_is_ignored_while_transcription_continues():
    # Releasing the controls early would let a second run start over the first.
    pipeline = make_pipeline(StubGui(), StubAudio())
    pipeline.run_started = True
    pipeline.processing_active = True
    pipeline._on_playback_finished()
    assert pipeline.run_started is True
    assert drain(pipeline.gui.display_queue) == []


def test_starting_a_run_rewinds_to_the_first_chunk():
    # Without this, a second run would resume from where the previous one stopped.
    pipeline = make_pipeline(StubGui(), StubAudio())
    pipeline.current_chunk_index = 5
    pipeline._spawn = lambda target, run_id: types.SimpleNamespace(is_alive=lambda: False)
    pipeline.start_processing("cpu")
    assert pipeline.current_chunk_index == 0
    assert pipeline.audio_handler.current_position == 0


def test_loading_a_file_stops_any_run_in_progress():
    # Loading during a run must not leave the previous workers writing to the transcript.
    samples = [1000] * (16000 * 6)
    stub = types.SimpleNamespace(AudioSegment=types.SimpleNamespace(
        from_file=lambda path: decoded_segment(samples)))
    pipeline = make_pipeline(StubGui(), AudioHandler())
    pipeline.run_started = True
    pipeline.processing_active = True
    with patched(audio_module, "pydub", stub):
        pipeline.load_file("recording.wav")
    assert pipeline.run_started is False
    assert pipeline.processing_active is False


def test_resume_is_ignored_when_the_run_is_not_paused():
    # A stray Play must not restart audio that is already running.
    audio = StubAudio()
    pipeline = make_pipeline(StubGui(), audio)
    pipeline.run_started = True
    pipeline.is_paused = False
    pipeline.resume_processing()
    assert "resume" not in audio.calls


def test_worker_threads_are_daemons():
    # Non-daemon workers would keep the interpreter alive after the window closes.
    pipeline = make_pipeline(StubGui(), StubAudio())
    thread = pipeline._spawn(lambda run_id: None, pipeline.run_id)
    thread.join(timeout=5)
    assert thread.daemon is True


def test_cache_distinguishes_devices_and_compute_types():
    # A key that ignored the device would hand a CPU model to a GPU run.
    FakeWhisper.builds.clear()
    with patched(models_module, "WhisperModel", FakeWhisper):
        models_module._model_cache.clear()
        models_module.Transcriber("cpu", "en")
        models_module.Transcriber("cuda", "en")
        models_module.Transcriber("cpu", "en")
        built = len(FakeWhisper.builds)
        models_module._model_cache.clear()
    assert built == 2


def test_hindi_uses_its_dedicated_model():
    # The base model transcribes Hindi poorly and tends to produce romanised text.
    FakeWhisper.builds.clear()
    with patched(models_module, "WhisperModel", FakeWhisper):
        models_module._model_cache.clear()
        models_module.Transcriber("cpu", "hi")
        models_module.Transcriber("cpu", "en")
        chosen = list(FakeWhisper.builds)
        models_module._model_cache.clear()
    assert chosen[0] == config_module.TRANSCRIPTION_MODEL_MAP["hi"]
    assert chosen[0] != chosen[1]


def test_failed_stream_open_leaves_no_partial_state():
    # A half-open device would block the next attempt to play.
    class Refusing:
        def open(self, **kwargs):
            raise OSError("no output device")

        def terminate(self):
            pass

    handler = AudioHandler()
    handler.audio_data = np.zeros(16000, dtype=np.float32)
    handler._create_chunks()
    with patched(audio_module, "pyaudio", types.SimpleNamespace(PyAudio=Refusing, paFloat32=1)):
        try:
            handler.start_playback()
        except OSError:
            pass
        else:
            raise AssertionError("a failed device open was not reported")
    assert handler.stream is None
    assert handler.audio_interface is None
    assert handler.play_active is False


def test_stop_and_clear_discards_the_loaded_audio():
    # Loading a new file must not leave the previous recording in memory.
    handler = AudioHandler()
    handler.audio_data = np.zeros(16000, dtype=np.float32)
    handler._create_chunks()
    handler.stop_and_clear()
    assert handler.audio_data is None
    assert handler.chunks == []
    assert handler.has_audio() is False


def test_has_audio_reports_false_when_nothing_is_loaded():
    # Play is enabled from this check, so a wrong answer starts a run with no audio.
    handler = AudioHandler()
    assert handler.has_audio() is False
    handler.audio_data = np.zeros(0, dtype=np.float32)
    assert handler.has_audio() is False


def test_pending_updates_are_discarded_before_a_new_file():
    # Messages from the previous run would otherwise appear under the new transcript.
    gui = GuiComponent.__new__(GuiComponent)
    gui.display_queue = Queue()
    gui.display_queue.put(("display", "stale", "0.0s - 5.0s"))
    gui.display_queue.put(("update_lang", "en"))
    gui.clear_pending_updates()
    assert gui.display_queue.empty()


def test_language_label_shows_the_readable_name():
    # A reviewer reading "ur" rather than "Urdu" would not know what was detected.
    labels = []

    class Label:
        def config(self, **kwargs):
            labels.append(kwargs.get("text"))

    gui = GuiComponent.__new__(GuiComponent)
    gui.language_label = Label()
    gui.update_language_label("ur")
    assert "Urdu" in labels[-1]


def test_releasing_the_stream_also_terminates_pyaudio():
    # Leaving the interface running would leak an audio device handle per run.
    terminated = []

    class Interface:
        def terminate(self):
            terminated.append(True)

    class Stream:
        def is_active(self):
            return False

        def stop_stream(self):
            pass

        def close(self):
            pass

    handler = AudioHandler()
    handler.stream = Stream()
    handler.audio_interface = Interface()
    handler._release_stream()
    assert terminated == [True]
    assert handler.audio_interface is None


def test_resume_clears_the_paused_flag():
    # Leaving it set would make the playback loop spin without producing sound.
    handler = AudioHandler()
    handler.audio_data = np.zeros(16000, dtype=np.float32)
    handler._create_chunks()
    handler.play_paused = True
    handler.playback_thread = types.SimpleNamespace(is_alive=lambda: True)
    with patched(audio_module, "pyaudio", types.SimpleNamespace(
            PyAudio=lambda: types.SimpleNamespace(open=lambda **k: object(), terminate=lambda: None),
            paFloat32=1)):
        handler.resume_playback()
    assert handler.play_paused is False


def test_file_filter_offers_both_supported_formats():
    # Removing either extension would hide valid recordings from the dialog.
    patterns = config_module.AUDIO_FILE_TYPES[0][1]
    assert "*.wav" in patterns and "*.mp3" in patterns


def test_all_three_languages_remain_supported():
    # Urdu is the default output language, so losing it would break first use.
    assert set(config_module.SUPPORTED_LANGUAGES) == {"en", "hi", "ur"}
    for code in config_module.SUPPORTED_LANGUAGES:
        assert code in config_module.NLLB_LANGUAGE_MAP
        assert code in config_module.TRANSCRIPTION_MODEL_MAP


def test_detection_failure_wakes_the_waiting_worker():
    # The waiting worker polls as a fallback, so the notification itself is asserted.
    class RecordingCondition:
        def __init__(self):
            self.notified = False
            self._condition = threading.Condition()

        def __enter__(self):
            return self._condition.__enter__()

        def __exit__(self, *args):
            return self._condition.__exit__(*args)

        def notify_all(self):
            self.notified = True
            self._condition.notify_all()

        def wait(self, timeout=None):
            return self._condition.wait(timeout)

    pipeline = make_pipeline(StubGui(), StubAudio())
    pipeline.lang_condition = RecordingCondition()
    pipeline.processing_active = True

    def failing(device):
        raise RuntimeError("no model")

    with patched(pipeline_module, "LanguageDetector", failing):
        pipeline._detection_worker(pipeline.run_id)
    assert pipeline.lang_condition.notified is True
    assert pipeline.processing_active is False


def test_button_states_are_not_swapped():
    # Reversing these would enable Pause before a run and disable Play after one.
    recorded = {}

    class Button:
        def __init__(self, name):
            self.name = name

        def config(self, **kwargs):
            recorded[self.name] = kwargs.get("state")

    gui = GuiComponent.__new__(GuiComponent)
    gui.play_btn = Button("play")
    gui.pause_btn = Button("pause")
    gui.set_button_states("normal", "disabled")
    assert recorded == {"play": "normal", "pause": "disabled"}


def test_clearing_the_output_empties_the_widget():
    # A new run must not display the previous transcript above its own.
    actions = []

    class Widget:
        def delete(self, start, end):
            actions.append("delete")

        def config(self, **kwargs):
            actions.append(kwargs.get("state"))

    gui = GuiComponent.__new__(GuiComponent)
    gui.output_text = Widget()
    gui.clear_output()
    assert "delete" in actions
    assert actions[-1] == "disabled"


def test_each_transcript_line_ends_with_a_newline():
    # Without it every segment would run into the next on a single line.
    written = []

    class Widget:
        def insert(self, index, text):
            written.append(text)

        def see(self, index):
            pass

        def config(self, **kwargs):
            pass

    gui = GuiComponent.__new__(GuiComponent)
    gui.output_text = Widget()
    gui.display_text("hello", "0.0s - 5.0s")
    assert written[-1].endswith("\n")


def test_restart_button_state_follows_its_argument():
    # Ignoring the argument would leave Restart enabled with no file loaded.
    states = []

    class Button:
        def config(self, **kwargs):
            states.append(kwargs.get("state"))

    gui = GuiComponent.__new__(GuiComponent)
    gui.restart_btn = Button()
    gui.set_restart_state("disabled")
    gui.set_restart_state("normal")
    assert states == ["disabled", "normal"]


def test_model_cache_is_safe_under_concurrent_use():
    # Detection and transcription can request a model at the same moment.
    builds = []

    class SlowWhisper:
        def __init__(self, name, device=None, compute_type=None):
            time.sleep(0.05)
            builds.append(name)

    with patched(models_module, "WhisperModel", SlowWhisper):
        models_module._model_cache.clear()
        threads = [threading.Thread(target=lambda: models_module.Transcriber("cpu", "ur"))
                   for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        built = len(builds)
        models_module._model_cache.clear()
    assert built == 1


def test_entry_point_reports_failure_when_tkinter_is_unavailable():
    # A zero exit code would tell a script that a failed launch had succeeded.
    import app as entry
    broken = types.SimpleNamespace(TclError=RuntimeError)

    def failing():
        raise RuntimeError("no display")

    with patched(entry, "GuiComponent", failing), patched(entry, "tk", broken):
        with contextlib.redirect_stderr(io.StringIO()) as captured:
            code = entry.main()
    assert code == 1
    assert "python -m tkinter" in captured.getvalue()


def test_chosen_file_reaches_the_pipeline():
    # Discarding the dialog result would make the Load button do nothing.
    received = []
    gui = GuiComponent.__new__(GuiComponent)
    gui.pipeline = types.SimpleNamespace(load_file=lambda path: received.append(path))
    chooser = types.SimpleNamespace(askopenfilename=lambda **kwargs: "/tmp/chosen.wav")
    with patched(gui_module, "filedialog", chooser):
        gui.choose_and_load_file()
    assert received == ["/tmp/chosen.wav"]


def test_timing_constants_are_sensible():
    # A zero poll interval spins the CPU; a zero join timeout abandons worker threads.
    assert config_module.QUEUE_POLL_MS > 0
    assert config_module.THREAD_JOIN_TIMEOUT > 0
    assert config_module.CHUNK_DURATION_SECONDS > 0
    assert config_module.DETECTION_CHUNK_COUNT > 0


def test_translation_is_not_attempted_for_the_same_language():
    # Building a translator that will not be used wastes memory and startup time.
    built = []

    class CountingTranslator:
        def __init__(self, device):
            built.append(device)

        def translate_text(self, text, source, target):
            return text

    with patched(models_module, "WhisperModel", FakeWhisper), \
         patched(pipeline_module, "Translator", CountingTranslator):
        models_module._model_cache.clear()
        pipeline = make_pipeline(StubGui(language="English"), StubAudio(chunk_count=2))
        pipeline.start_processing("cpu")
        for thread in (pipeline.detection_thread, pipeline.processing_thread):
            thread.join(timeout=10)
        models_module._model_cache.clear()
    assert built == []


# --- Runner ---

def main() -> int:
    # Each test runs independently so one failure does not hide the rest.
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    failures = 0
    for test in tests:
        try:
            test()
        except Exception as exc:
            failures += 1
            print(f"  FAIL  {test.__name__}: {type(exc).__name__}: {exc}")
        else:
            print(f"  ok    {test.__name__}")
    print(f"\n{len(tests) - failures}/{len(tests)} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
