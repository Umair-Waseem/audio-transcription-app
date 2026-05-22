# 🎤 Offline Real-Time Audio Transcription App

This project is a Tkinter desktop app for offline audio transcription and translation. It loads a WAV or MP3 file, plays it back, detects the spoken language, transcribes the audio with Faster Whisper, optionally translates the transcript with a local NLLB CTranslate2 model, displays timestamped text, and exports the result as a `.txt` file.

---

## 🚀 What You Can Do

* 🔌 **Work Offline**: No internet needed after setup
* 🧠 **Detect Language Automatically**: Supports English, Urdu, and Hindi
* 📝 **Transcribe Audio in Real Time**: Converts speech into text with timestamps
* 🌐 **Translate Transcripts**: Choose your output language
* 🎧 **Listen While You Read**: Built-in audio playback
* 💾 **Export Your Work**: Save the transcript as a `.txt` file
* 🖥️ **Easy to Use**: Clean and beginner-friendly interface

---

## 💼 Best For

* Transcribing lectures, interviews, or meetings
* Creating multilingual content
* Assisting people with hearing difficulties
* Language learning and study support

---


## Project Structure

```text
audio-transcription-app/
|-- app.py
|-- requirements.txt
|-- README.md
|-- assets/
|   `-- OSR_in_000_0063_8k.wav
`-- models/                         # Required for translation, not committed
    |-- flores200_sacrebleu_tokenizer_spm.model
    `-- nllb-200-distilled-600M-ct2/
```

## Supported Languages

| Language | Code |
| --- | --- |
| English | `en` |
| Hindi | `hi` |
| Urdu | `ur` |

## Requirements

- Python 3.10-3.12 recommended. On Windows, use Python 3.12 for the most reliable dependency installation.
- FFmpeg on your system `PATH` for MP3 support through `pydub`.
- Local translation model files under `models/` if you want translated output.
- On Windows, Python 3.14 can force native builds for packages such as `pyaudio`, which requires Microsoft C++ Build Tools and PortAudio headers. Recreate the environment with Python 3.12 instead of building these packages manually.
- Python 3.13 and newer also require `audioop-lts`, which is included in `requirements.txt`, because the standard-library `audioop` module was removed.

## Setup

```powershell
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If the `py` launcher is unavailable, run the Python 3.12 executable directly to create the environment, then activate it with the same command above.

Confirm the environment is using Python 3.12 before installing:

```powershell
python --version
```

Confirm Tkinter is available for the desktop GUI:

```powershell
python -m tkinter
```

If that command fails with a Tcl/Tk error, repair or reinstall Python 3.12 and make sure the Tcl/Tk and IDLE option is selected.

If you already created the virtual environment and see `ModuleNotFoundError: No module named 'audioop'` or `No module named 'pyaudioop'`, reinstall the requirements:

```powershell
python -m pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source venv/bin/activate
```

For MP3 support on Windows, install FFmpeg and reopen PowerShell:

```powershell
winget install -e --id Gyan.FFmpeg
ffmpeg -version
```

## Translation Model Setup

The app can transcribe without the local NLLB translation model when the selected output language matches the detected source language. Translation between languages requires these files:

1. Convert `facebook/nllb-200-distilled-600M` to CTranslate2 format:

```bash
python -m pip install transformers==4.41.1
ct2-transformers-converter \
  --model facebook/nllb-200-distilled-600M \
  --output_dir models/nllb-200-distilled-600M-ct2 \
  --quantization int8
```

2. Download the FLORES tokenizer and save it as:

```text
models/flores200_sacrebleu_tokenizer_spm.model
```

## Run

```powershell
python app.py
```

Then load an audio file, choose CPU or GPU, choose the desired output language, and press Play. Use Restart after changing the output language to clear the previous text and process the loaded audio again with the newly selected language.

## Test

There is no automated test suite in this repository yet. Use these checks after changes:

```powershell
python -m py_compile app.py
python app.py
```

## Notes

- Whisper models are downloaded by Faster Whisper when first used, so the first run may need internet access unless the models are already cached.
- Translation model files are intentionally not included because they are large.
- Exported transcripts are written to the current user's Downloads folder.

---

## 🪪 License

This project is licensed under the **MIT License**. 

---

## 🙏 Thanks To

* OpenAI — Whisper
* Facebook AI — NLLB
* OpenNMT — CTranslate2
* Google — SentencePiece
* Hugging Face — Model hosting

---

## 🔧 What’s Coming Next

* 🎙️ Live microphone transcription
* 🌐 More language support
* 🌙 Dark mode
* 🧩 Select models from within the app

---

## 🤝 Want to Help?

You’re welcome to contribute! Fork the project, make your changes, and open a pull request.

---

## 🖼️ Screenshots

<img width="1360" height="672" alt="image" src="https://github.com/user-attachments/assets/082d0a71-a9a4-4296-ad8c-8556bb3655be" />

---

## 📬 Contact

Have questions or suggestions?

* Open an issue on GitHub
* Email: `umairwaseem5.4.2003@gmail.com`

> Made with ❤️ using Python — to help you turn your audio into accurate, readable text, anytime.
