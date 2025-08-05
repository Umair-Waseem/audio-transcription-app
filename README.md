# 🎤 Offline Real-Time Audio Transcription App

This is a simple and helpful desktop app that lets you **convert audio into text and translate it — all offline**. It runs entirely on your computer and uses powerful tools like [Whisper](https://github.com/openai/whisper), [CTranslate2](https://github.com/OpenNMT/CTranslate2), and [Tkinter](https://docs.python.org/3/library/tkinter.html) for its interface.

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

## ⚙️ How to Get Started

### 1. Download the Project

```bash
git clone https://github.com/Umair-Waseem/audio-transcription-app.git
cd audio-transcription-app
```

### 2. (Optional) Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install the Requirements

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the App

```bash
python app.py
```

The app window will open. Load your audio file, and the app will handle the rest!

---

## 📁 Project Folder Structure

```
audio-transcription-app/
├── app.py                       
├── requirements.txt            
├── README.md                                        
├── models/                     
│   ├── flores200_sacrebleu_tokenizer_spm.model         
│   └── nllb-200-distilled-600M-ct2/                   
├── assets/                                    
```

---

## 🌍 Supported Languages

| Language | Code |
| -------- | ---- |
| English  | `en` |
| Urdu     | `ur` |
| Hindi    | `hi` |

---

## 🔄 How It Works

1. Load your `.mp3` or `.wav` audio file
2. The app detects the language automatically
3. It transcribes the speech into text
4. It translates the text if needed
5. You see timestamped output in the interface
6. You can export the final transcript as `.txt`

---

## 📦 Required Libraries

Install all libraries with:

```bash
pip install -r requirements.txt
```

Main packages:

* `faster-whisper`
* `ctranslate2`
* `sentencepiece`
* `pyaudio`
* `pydub`
* `numpy==1.26.4`
* `tk` 
* `transformers==4.41.1` 
---

## 🧠 AI Models Used

### 🔊 Transcription Models (Auto-Downloaded)

These models are automatically downloaded by the program when needed:

* `whisper-base` — for English
* `vasista22/whisper-hindi-small` — for Hindi
* `tiny` — for language detection

> ✅ You do not need to manually download or convert these models.

### 🌐 Translation and Tokenization Models (Manual Setup Required)

Due to GitHub's file size limits, the following models must be **downloaded and set up manually**:

#### 1. **Translation Model:** `nllb-200-distilled-600M-ct2`

* Download from Hugging Face: [facebook/nllb-200-distilled-600M](https://huggingface.co/facebook/nllb-200-distilled-600M)
* Convert to CTranslate2 format using the following command:

```bash
ct2-transformers-converter \
  --model facebook/nllb-200-distilled-600M \
  --output_dir models/nllb-200-distilled-600M-ct2 \
  --quantization int8
```

* Place the converted folder inside the `models/` directory.

#### 2. **Tokenizer Model:** `flores200_sacrebleu_tokenizer_spm.model`

* Download from: [flores200 tokenizer](https://dl.fbaipublicfiles.com/nllb/flores200/sacrebleu_tokenizer_spm.model)
* Save it in the `models/` directory as shown in the structure above.

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

## ❓ Troubleshooting

* **App won’t open?** Make sure you have Python 3.8+ and all dependencies installed
* **No sound?** Check that `pyaudio` is installed and your audio device is working
* **No transcription?** Confirm models are properly placed and the audio is clear

---

## 🤝 Want to Help?

You’re welcome to contribute! Fork the project, make your changes, and open a pull request.

---

## 🖼️ Screenshots

*Add your screenshots here to showcase the app interface.*

---

## 📬 Contact

Have questions or suggestions?

* Open an issue on GitHub
* Email: `umairwaseem5.4.2003@gmail.com`

> Made with ❤️ using Python — to help you turn your audio into accurate, readable text, anytime.
