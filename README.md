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

## Deploy to Hugging Face Spaces (Gradio SDK)

Include these files at the repo root:

- app.py
- keyboard.html
- requirements.txt

Then create a Gradio Space and push the repo.

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
