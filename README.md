---
title: Virtual Keyboard
emoji: 🎹
colorFrom: red
colorTo: gray
sdk: gradio
sdk_version: 6.5.1
app_file: app.py
pinned: false
short_description: A small virtual midi keyboard
---

# Virtual MIDI Keyboard

Minimal browser MIDI keyboard: play in the browser, record note events, export a .mid file.

## Features

- 🎹 Two-octave virtual piano keyboard
- 🎵 Multiple instrument sounds (Synth, Piano, Organ, Bass, Pluck, FM)
- ⌨️ Computer keyboard input support  
- 📹 MIDI event recording with timestamps
- 💾 Export recordings as .mid files
- 📊 Real-time MIDI event monitor
- 🎨 Clean, responsive interface

## Project Structure

```
virtual_keyboard/
├── app.py              # Gradio server + MIDI conversion
├── keyboard.html       # Main UI structure
├── static/
│   ├── styles.css      # All application styles
│   ├── keyboard.js     # Client-side logic
│   └── README.md       # Static assets documentation
├── requirements.txt    # Python dependencies
├── pyproject.toml      # Project metadata
└── README.md           # This file
```

## Files

- app.py: Gradio app + MIDI export API
- keyboard.html: client-side keyboard (Tone.js)

## Run locally

```bash
uv venv
uv pip install -r requirements.txt
uv run python app.py
```

Open http://127.0.0.1:7860

## Deploy to Hugging Face Spaces

### Quick Setup

1. **Create a Space**
   - Go to https://huggingface.co/spaces
   - Click "Create new Space"
   - Choose **Gradio SDK**
   - Name it (e.g., `virtual_keyboard`)

2. **Add HF remote and push**
   ```bash
   git remote add hf git@hf.co:spaces/YOUR_USERNAME/virtual_keyboard
   git push hf main
   ```

That's it! Your Space will automatically deploy.

### Push to Both GitHub and HF

```bash
git push origin main && git push hf main
```

## API

The browser posts events to the Gradio call endpoint:

```
POST /gradio_api/call/save_midi
{
  "data": [events]
}
```

The response returns an event_id. Fetch the result from:

```
GET /gradio_api/call/save_midi/{event_id}
```

The response includes base64 MIDI data at data[0].midi_base64.
