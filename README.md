# Offline Audio Transcription and Translation

A Tkinter desktop application that transcribes and translates local audio files entirely on your
own machine. It loads a WAV or MP3 file, plays it back, detects the spoken language, transcribes it
with Faster-Whisper, optionally translates the result with a local NLLB-200 model, shows timestamped
text, and exports the transcript.

Supported languages: **English**, **Urdu**, **Hindi**.

---

## Features

- **Runs on your machine.** All inference is local. Only the first run needs the internet, to
  download the Whisper model, and only if it is not already cached.
- **Automatic language detection** from the opening seconds of the file.
- **Timestamped transcription** in five-second segments, with voice-activity filtering so silence
  is not transcribed.
- **Optional translation** between any two supported languages, using a locally converted,
  int8-quantised NLLB-200 model.
- **Synchronised playback** while the transcript builds, with pause and resume.
- **Right-to-left rendering for Urdu** when a Nastaliq font is installed.
- **Export** the transcript to a location you choose.
- **CPU or GPU**, selectable at runtime.

---

## How it works

| Stage | What happens |
| --- | --- |
| Load | Audio is resampled to 16 kHz mono and normalised to the range models expect |
| Segment | The signal is split into five-second chunks with no overlap |
| Detect | The first three chunks are used to identify the spoken language |
| Transcribe | Each chunk is transcribed with a model chosen for that language |
| Translate | If the output language differs from the source, text passes through NLLB-200 |
| Display | Results appear with timestamps as each chunk completes |

Detection, transcription and playback run on separate threads, so the interface stays responsive.
All widget updates are marshalled through a queue, so no worker thread touches Tkinter directly.

Models are loaded once and reused, so pressing **Restart** does not pay the loading cost again.

Hindi is transcribed with a fine-tuned checkpoint rather than the base model, because the base model
performs poorly on Hindi and tends to produce romanised output.

---

## Requirements

- Python 3.10 or newer. On Windows, 3.12 installs the binary dependencies most reliably;
  3.13 and later additionally need `audioop-lts`, which `requirements.txt` installs automatically.
- FFmpeg on your `PATH` for MP3 support.
- Java is **not** required.
- Local NLLB model files only if you want translation between different languages.

---

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: .\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Confirm Tkinter is available:

```bash
python -m tkinter
```

If that fails, reinstall Python with the Tcl/Tk option selected.

For MP3 support on Windows:

```powershell
winget install -e --id Gyan.FFmpeg
```

---

## Translation model setup

Transcription works without these files. Translation between two different languages needs them.

```bash
python -m pip install transformers==4.41.1
ct2-transformers-converter \
  --model facebook/nllb-200-distilled-600M \
  --output_dir models/nllb-200-distilled-600M-ct2 \
  --quantization int8
```

Then place the FLORES tokenizer at `models/flores200_sacrebleu_tokenizer_spm.model`.

The application checks for both before translating and reports exactly which file is missing.

---

## Running

```bash
python app.py
```

Load a file, choose CPU or GPU, choose the output language, then press **Play**. Use **Restart**
after changing the output language to reprocess the same file.

---

## Project structure

```text
audio-transcription-app/
├── app.py                      # Entry point
├── audio_transcription/
│   ├── config.py               # Constants, language maps, model paths
│   ├── audio.py                # Decoding, chunking, playback
│   ├── models.py               # Model cache, detection, transcription, translation
│   ├── pipeline.py             # Threading and orchestration
│   └── gui.py                  # Tkinter interface
├── requirements.txt
├── LICENSE
├── assets/                     # Sample audio
└── models/                     # Translation model files, not committed
```

---

## Known limitations

- Live microphone input is not supported; the application reads files from disk.
- Language detection covers only English, Urdu and Hindi. Other languages are rejected with a
  clear message rather than transcribed incorrectly.
- Chunks are cut at fixed five-second boundaries, so a word spanning a boundary may be split.
- Translation quality depends on the NLLB-200 distilled model and is not reviewed by the app.

---

## Licence

MIT. See [LICENSE](LICENSE).

---

## Acknowledgements

OpenAI (Whisper) · Meta AI (NLLB) · OpenNMT (CTranslate2) · Google (SentencePiece) · Hugging Face
