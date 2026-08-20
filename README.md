# Offline Audio Transcription and Translation

A desktop application that transcribes and translates recorded audio entirely on the machine it
runs on. It reads a local WAV or MP3 file and identifies the spoken language, then transcribes the
speech with Faster-Whisper. The result can be translated with a local NLLB-200 model. Timestamped
text appears as each segment completes.

Supported languages: **English**, **Urdu** and **Hindi**.

No audio or text is sent to a remote service. The only network access the application requires is
the initial download of the Whisper model, and only if it is not already cached.

---

## Contents

- [Features](#features)
- [How it works](#how-it-works)
- [Requirements](#requirements)
- [Installation](#installation)
- [Translation model setup](#translation-model-setup)
- [Usage](#usage)
- [Project structure](#project-structure)
- [Tests](#tests)
- [Known limitations](#known-limitations)
- [Licence](#licence)
- [Acknowledgements](#acknowledgements)

---

## Features

**Local processing.** Transcription and translation run on the machine. Nothing is uploaded.

**Automatic language detection.** The spoken language is identified from the opening seconds of the
recording. Languages outside the supported set are reported rather than transcribed incorrectly.

**Timestamped transcription.** Audio is transcribed in five-second segments, each labelled with its
position in the recording. Voice-activity filtering prevents silence from being transcribed.

**Optional translation.** Output can be produced in any supported language. Translation uses a
locally converted, int8-quantised NLLB-200 model and is skipped when the source and output
languages match.

**Synchronised playback.** The recording plays while the transcript builds, with pause and resume.

**Right-to-left rendering.** Urdu is displayed in a Nastaliq font when one is installed, and in a
readable fallback otherwise.

**Transcript export.** The transcript can be saved to a location of your choosing, encoded as UTF-8.

**Selectable compute device.** CPU or CUDA, chosen at runtime.

---

## How it works

| Stage | Behaviour |
| --- | --- |
| Load | The file is decoded, resampled to 16 kHz mono, and normalised to the range the models expect |
| Segment | The signal is divided into five-second segments with no overlap |
| Detect | The first three segments are used to identify the spoken language |
| Transcribe | Each segment is transcribed using the model selected for that language |
| Translate | If the output language differs from the source, the text passes through NLLB-200 |
| Display | Each result appears with its timestamp as soon as it is ready |

Detection, transcription and playback run on separate threads, so the interface remains responsive
throughout. Worker threads never modify widgets directly; every update is placed on a queue that
the interface drains on its own thread.

Models are loaded once and reused, so pressing **Restart** does not repeat the loading cost.

Hindi is transcribed with a fine-tuned checkpoint rather than the base model, which transcribes
Hindi poorly and tends to produce romanised output.

---

## Requirements

| Requirement | Notes |
| --- | --- |
| Python 3.10 or newer | On Windows, 3.12 installs the binary dependencies most reliably |
| Tcl/Tk | Included with most Python distributions; verify with `python -m tkinter` |
| PortAudio | Needed on Linux before PyAudio will build; bundled in the Windows and macOS wheels |
| FFmpeg | Required for MP3 input only; WAV files need no additional software |
| NLLB-200 model files | Required for translation only; transcription works without them |

Java is not required.

On Debian and Ubuntu, install the audio and Tk system packages before the Python dependencies.

```bash
sudo apt-get install portaudio19-dev python3-tk
```

Without PortAudio, installing PyAudio fails while building its wheel.

---

## Installation

Create and activate a virtual environment, then install the dependencies.

```bash
python -m venv venv
source venv/bin/activate          # Windows: .\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Confirm that Tkinter is available.

```bash
python -m tkinter
```

If this fails, reinstall Python with the Tcl/Tk option selected.

To enable MP3 input on Windows, install FFmpeg and ensure it is on your `PATH`.

```powershell
winget install -e --id Gyan.FFmpeg
```

---

## Translation model setup

This section is required only if you intend to translate between two different languages.
Transcription works without it.

Convert the NLLB-200 distilled model to the CTranslate2 format.

```bash
python -m pip install transformers==4.41.1
ct2-transformers-converter \
  --model facebook/nllb-200-distilled-600M \
  --output_dir models/nllb-200-distilled-600M-ct2 \
  --quantization int8
```

Place the FLORES tokenizer at `models/flores200_sacrebleu_tokenizer_spm.model`.

The application verifies both paths before translating and reports precisely which file is missing.

---

## Usage

```bash
python app.py
```

1. Select **Load File** and choose a WAV or MP3 recording.
2. Choose **CPU** or **GPU**.
3. Choose the output language.
4. Select **Play**.

The transcript appears segment by segment as the recording plays. **Pause** suspends both playback
and transcription; **Play** resumes them. **Restart** reprocesses the same file, which is the way to
apply a different output language. **Export** saves the transcript to a file.

The output language cannot be changed while a run is in progress, because the choice is applied when
the run begins.

---

## Project structure

```text
audio-transcription-app/
├── app.py                      # Entry point
├── audio_transcription/
│   ├── config.py               # Constants, language maps and model paths
│   ├── audio.py                # Decoding, segmentation and playback
│   ├── models.py               # Model cache, detection, transcription and translation
│   ├── pipeline.py             # Thread coordination and run control
│   └── gui.py                  # Tkinter interface
├── tests/
│   └── test_app.py             # Test suite
├── assets/                     # Sample recording
├── models/                     # Translation model files, not tracked in version control
├── requirements.txt
└── LICENSE
```

Only `gui.py` imports Tkinter. The configuration, audio, model and pipeline layers can be imported
and tested without a display, which is what allows the test suite to run in a headless environment.

---

## Tests

```bash
python tests/test_app.py
```

Fifty-eight tests cover the behaviour the application depends on. They verify segmentation and
timestamp alignment, sample normalisation, playback state and failure handling, model caching and
selection, and transcription parameters. They also cover language detection and its rejection
cases, file loading, display-queue resilience, control state and locking, and the entry point.
The remainder address shutdown ordering, transcript export and encoding, translation tagging,
pause and resume, restart, run invalidation, and thread lifecycle across repeated runs.

Each test runs independently and restores anything it modifies. A single failure therefore neither
hides the remaining results nor affects them.

The suite has been checked by mutation testing. Forty-nine defects were introduced into the source
one at a time, and every one was detected. The tests require no audio device, no display and no
model files, and the suite exits with a non-zero status if any test fails.

---

## Known limitations

**Recorded files only.** The application reads audio from disk. Live microphone input is not
supported.

**Three languages.** Detection covers English, Urdu and Hindi. Other languages are reported as
unsupported rather than transcribed inaccurately.

**Fixed segment boundaries.** Segments are cut at five-second intervals, so a word spanning a
boundary may be divided between two lines.

**Translation quality.** Output quality is determined by the NLLB-200 distilled model and is not
assessed by the application.

---

## Licence

Released under the MIT Licence. See [LICENSE](LICENSE).

---

## Acknowledgements

This project builds on OpenAI's Whisper, Meta AI's No Language Left Behind, OpenNMT's CTranslate2,
Google's SentencePiece, and the Faster-Whisper implementation from SYSTRAN.
