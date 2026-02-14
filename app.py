"""
Virtual MIDI Keyboard - Gradio Application

A browser-based MIDI keyboard that can:
- Play notes with various synthesizer sounds
- Record MIDI events with timestamps
- Export recordings as .mid files
- Support computer keyboard input
- Monitor MIDI events in real-time
"""

import base64
import html
import gradio as gr

from config import INSTRUMENTS, KEYBOARD_KEYS, KEYBOARD_SHORTCUTS
from midi import events_to_midbytes
from engines import EngineRegistry


# =============================================================================
# API ENDPOINTS
# =============================================================================


def save_midi_api(events):
    """Export recorded MIDI events to .mid file"""
    if not isinstance(events, list) or len(events) == 0:
        return {"error": "No events provided"}

    mid_bytes = events_to_midbytes(events)
    midi_b64 = base64.b64encode(mid_bytes).decode("ascii")
    return {"midi_base64": midi_b64}


def get_config():
    """Provide frontend with instruments and keyboard layout"""
    return {
        "instruments": INSTRUMENTS,
        "keyboard_keys": KEYBOARD_KEYS,
        "keyboard_shortcuts": KEYBOARD_SHORTCUTS,
        "engines": [
            {"id": engine_id, "name": EngineRegistry.get_engine_info(engine_id)["name"]}
            for engine_id in EngineRegistry.list_engines()
        ],
    }


def process_with_engine(engine_id: str, events: list):
    """Process MIDI events through selected engine"""
    if not engine_id or not events:
        return {"error": "Missing engine_id or events"}

    try:
        engine = EngineRegistry.get_engine(engine_id)
        processed = engine.process(events)
        return {"success": True, "events": processed}
    except ValueError as e:
        return {"error": str(e)}
    except Exception as e:
        return {"error": f"Processing error: {str(e)}"}


# =============================================================================
# GRADIO UI
# =============================================================================

# Load HTML and CSS
with open("keyboard.html", "r", encoding="utf-8") as f:
    html_content = f.read()

with open("static/styles.css", "r", encoding="utf-8") as f:
    css_content = f.read()

with open("static/keyboard.js", "r", encoding="utf-8") as f:
    js_content = f.read()

# Inject CSS and JS into HTML
keyboard_html = html_content.replace(
    '<link rel="stylesheet" href="/file=static/styles.css" />',
    f"<style>{css_content}</style>",
).replace(
    '<script src="/file=static/keyboard.js"></script>', f"<script>{js_content}</script>"
)

iframe_html = (
    '<iframe srcdoc="'
    + html.escape(keyboard_html, quote=True)
    + '" style="width:100%;height:750px;border:0;"></iframe>'
)

# Create Gradio app
with gr.Blocks(title="Virtual MIDI Keyboard") as demo:
    gr.HTML(iframe_html)

    # Hidden config API
    with gr.Group(visible=False):
        config_input = gr.Textbox(label="_")
        config_output = gr.JSON(label="_")
        config_btn = gr.Button("get_config", visible=False)
        config_btn.click(
            fn=lambda x: get_config(),
            inputs=config_input,
            outputs=config_output,
            api_name="config",
        )

        # MIDI save API
        midi_input = gr.JSON(label="_")
        midi_output = gr.JSON(label="_")
        midi_btn = gr.Button("save_midi", visible=False)
        midi_btn.click(
            fn=save_midi_api,
            inputs=midi_input,
            outputs=midi_output,
            api_name="save_midi",
        )

        # Engine processing API
        engine_input = gr.JSON(label="_")
        engine_output = gr.JSON(label="_")
        engine_btn = gr.Button("process_engine", visible=False)
        engine_btn.click(
            fn=lambda payload: process_with_engine(
                payload.get("engine_id"), payload.get("events", [])
            ),
            inputs=engine_input,
            outputs=engine_output,
            api_name="process_engine",
        )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    demo.launch()
