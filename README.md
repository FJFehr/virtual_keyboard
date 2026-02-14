---
title: SYNTHIA
emoji: 🎹
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
pinned: false
short_description: Browser-based MIDI keyboard with recording and synthesis
---

# SYNTHIA

A minimal, responsive browser-based MIDI keyboard. Play live, record performances, and export as MIDI files. 🎹


## 🗂️ Project Structure

```
.
├── app.py                  # Gradio server & API endpoints
├── config.py               # Centralized configuration
├── engines.py              # MIDI processing engines
├── midi.py                 # MIDI file utilities
├── keyboard.html           # HTML structure
├── static/
│   ├── keyboard.js         # Client-side audio (Tone.js)
│   └── styles.css          # Styling & animations
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## 🚀 Quick Start

```bash
# Install dependencies
uv pip install -r requirements.txt

# Run the app
uv run python app.py
```

Open **http://127.0.0.1:7861**

## 🌐 Deploy to Hugging Face Spaces

```bash
git remote add hf git@hf.co:spaces/YOUR_USERNAME/synthia
git push hf main
```

## 🔧 Technology

- **Frontend**: Tone.js v6+ (Web Audio API)
- **Backend**: Gradio 6.x + Python 3.10+
- **MIDI**: mido library

## 📝 License

Open source - free to use and modify.

