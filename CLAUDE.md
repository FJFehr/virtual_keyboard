# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
uv sync

# Run the app
uv run python app.py
```

App serves at **http://127.0.0.1:7860**

Set `HF_TOKEN` environment variable to authenticate with Hugging Face (required for Godzilla model download).

## Architecture

**SYNTHIA** is a browser-based MIDI keyboard with AI continuation. Three layers:

- **Backend** (`app.py`, `engines.py`, `midi_model.py`, `midi.py`, `config.py`): Python/Gradio
- **Frontend** (`static/keyboard.js`, `static/styles.css`, `keyboard.html`): Tone.js audio synthesis
- **Bridge**: Hidden Gradio components (textboxes + buttons with `elem_id`s) act as the API layer between JS and Python

### Frontend → Backend Communication Pattern

JavaScript communicates with Python by:
1. Writing a JSON string into a hidden `gr.Textbox` (`elem_id="vk_engine_input"`, etc.)
2. Programmatically clicking a hidden `gr.Button` (`elem_id="vk_engine_cpu_btn"` or `"vk_engine_gpu_btn"`)
3. Reading the JSON response from a hidden output `gr.Textbox`

This pattern is used for all three bridges: `get_config`, `save_midi`, and `process_engine` (CPU and GPU variants).

### Engine System

Engines live in `engines.py` and are registered in `EngineRegistry._engines`. Each engine class has a `process(events, options, request, device)` method that takes/returns MIDI event dicts with keys: `type`, `note`, `velocity`, `time`, `channel`.

Current engines: `parrot`, `reverse_parrot`, `godzilla_continue`.

To add a new engine: create a class in `engines.py`, register it in `EngineRegistry._engines`, and add a tooltip in `keyboard.js` (`populateEngineSelect()`).

### GPU vs CPU

`process_engine_event_bridge_gpu` is decorated with `@spaces.GPU(duration=120)` for Hugging Face Spaces. The `device` parameter (`"cuda"` or `"cpu"`) is forwarded all the way to the model inference.

### Godzilla Model

`midi_model.py` handles the Hugging Face transformer model. It's preloaded in a background daemon thread at startup (`start_background_preload()`). `get_model("godzilla")` returns a cached singleton. The model tokenizes MIDI events, runs autoregressive generation, then detokenizes back to events.

### Config as Single Source of Truth

`config.py` defines `INSTRUMENTS`, `KEYBOARD_KEYS`, and `KEYBOARD_SHORTCUTS`. These are served to the frontend via the `get_config` bridge endpoint — the frontend does not hardcode them.
