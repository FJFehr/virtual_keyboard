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
