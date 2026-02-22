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
import json
import os
import re
from threading import Thread
import traceback
import gradio as gr
from huggingface_hub import login

from config import INSTRUMENTS, KEYBOARD_KEYS, KEYBOARD_SHORTCUTS
from midi import events_to_midbytes
from midi_model import preload_godzilla_assets, preload_godzilla_model
from engines import EngineRegistry
import spaces


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


def _parse_json_payload(payload_text: str | None, default):
    if payload_text is None or payload_text == "":
        return default
    try:
        return json.loads(payload_text)
    except json.JSONDecodeError:
        return default


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

def process_with_engine(
    engine_id: str,
    events: list,
    options: dict | None = None,
    request: "gr.Request | None" = None,
    device: str = "auto",
):
    """Process MIDI events through selected engine"""
    if not engine_id or not events:
        return {"error": "Missing engine_id or events"}

    x_ip_token = (
        request.headers.get("x-ip-token")
        if request is not None and hasattr(request, "headers")
        else None
    )
    print(
        "process_engine auth:",
        {
            "engine_id": engine_id,
            "has_x_ip_token": bool(x_ip_token),
        },
    )

    try:
        engine = EngineRegistry.get_engine(engine_id)
        processed = engine.process(
            events,
            options=options,
            request=request,
            device=device,
        )
        return {"success": True, "events": processed}
    except ValueError as e:
        traceback.print_exc()
        return {"error": str(e)}
    except Exception as e:
        traceback.print_exc()
        return {"error": f"Processing error: {str(e)}"}


def process_engine_payload(
    payload: dict,
    request: "gr.Request | None" = None,
    device: str = "auto",
):
    if not isinstance(payload, dict):
        return {"error": "Invalid payload"}
    return process_with_engine(
        payload.get("engine_id"),
        payload.get("events", []),
        options=payload.get("options"),
        request=request,
        device=device,
    )


def config_event_bridge(_payload_text: str) -> str:
    return json.dumps(get_config())


def save_midi_event_bridge(payload_text: str) -> str:
    events = _parse_json_payload(payload_text, [])
    result = save_midi_api(events)
    return json.dumps(result)


def process_engine_event_bridge_cpu(
    payload_text: str,
    request: "gr.Request | None" = None,
) -> str:
    payload = _parse_json_payload(payload_text, {})
    result = process_engine_payload(payload, request=request, device="cpu")
    return json.dumps(result)


@spaces.GPU(duration=120)
def process_engine_event_bridge_gpu(
    payload_text: str,
    request: "gr.Request | None" = None,
) -> str:
    payload = _parse_json_payload(payload_text, {})
    result = process_engine_payload(payload, request=request, device="cuda")
    return json.dumps(result)


def start_background_preload() -> None:
    def _run() -> None:
        try:
            checkpoint_path = preload_godzilla_assets()
            print(f"Godzilla assets preloaded: {checkpoint_path}")
            model_info = preload_godzilla_model(device="cpu")
            print(f"Godzilla model warmed in memory: {model_info}")
        except Exception:
            print("Godzilla preload failed:")
            traceback.print_exc()

    Thread(target=_run, daemon=True).start()


def login_huggingface_from_env() -> None:
    """Authenticate with Hugging Face if HF_TOKEN is available."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set; skipping huggingface_hub.login()")
        return

    try:
        login(token=token, add_to_git_credential=False)
        print("Authenticated with Hugging Face using HF_TOKEN")
    except Exception:
        print("huggingface_hub login failed:")
        traceback.print_exc()


# =============================================================================
# GRADIO UI
# =============================================================================

login_huggingface_from_env()
start_background_preload()

# Load HTML and CSS
with open("keyboard.html", "r", encoding="utf-8") as f:
    html_content = f.read()

with open("static/styles.css", "r", encoding="utf-8") as f:
    css_content = f.read()

with open("static/keyboard.js", "r", encoding="utf-8") as f:
    js_content = f.read()

body_match = re.search(r"<body[^>]*>(.*)</body>", html_content, flags=re.IGNORECASE | re.DOTALL)
keyboard_markup = body_match.group(1) if body_match else html_content
keyboard_markup = re.sub(r"<script\b[^>]*>.*?</script>", "", keyboard_markup, flags=re.IGNORECASE | re.DOTALL)
keyboard_markup = re.sub(r"<link\b[^>]*>", "", keyboard_markup, flags=re.IGNORECASE)

# Make logo rendering robust by embedding local repo logo bytes directly.
logo_path = "synthia_logo.png"
if os.path.exists(logo_path):
    try:
        with open(logo_path, "rb") as logo_file:
            logo_b64 = base64.b64encode(logo_file.read()).decode("ascii")
        keyboard_markup = keyboard_markup.replace(
            'src="/file=synthia_logo.png"',
            f'src="data:image/png;base64,{logo_b64}"',
        )
    except Exception:
        print("Failed to embed synthia_logo.png; keeping original src path.")
        traceback.print_exc()
else:
    print("synthia_logo.png not found; logo image may not render.")

hidden_bridge_css = "\n.vk-hidden { display: none !important; }\n"
head_html = "\n".join(
    [
        '<script src="https://unpkg.com/tone@next/build/Tone.js"></script>',
        f"<script>{js_content}</script>",
    ]
)

# Create Gradio app
with gr.Blocks(title="Virtual MIDI Keyboard", css=css_content + hidden_bridge_css, head=head_html) as demo:
    gr.HTML(keyboard_markup)

    # Hidden bridges for direct Gradio event calls from frontend JS
    with gr.Group(elem_classes=["vk-hidden"]):
        config_input = gr.Textbox(value="{}", elem_id="vk_config_input", show_label=False)
        config_output = gr.Textbox(elem_id="vk_config_output", show_label=False)
        config_btn = gr.Button("get_config", elem_id="vk_config_btn")
        config_btn.click(
            fn=config_event_bridge,
            inputs=config_input,
            outputs=config_output,
        )

        midi_input = gr.Textbox(value="[]", elem_id="vk_save_input", show_label=False)
        midi_output = gr.Textbox(elem_id="vk_save_output", show_label=False)
        midi_btn = gr.Button("save_midi", elem_id="vk_save_btn")
        midi_btn.click(
            fn=save_midi_event_bridge,
            inputs=midi_input,
            outputs=midi_output,
        )

        engine_input = gr.Textbox(value="{}", elem_id="vk_engine_input", show_label=False)

        engine_cpu_output = gr.Textbox(elem_id="vk_engine_cpu_output", show_label=False)
        engine_cpu_btn = gr.Button("process_engine_cpu", elem_id="vk_engine_cpu_btn")
        engine_cpu_btn.click(
            fn=process_engine_event_bridge_cpu,
            inputs=engine_input,
            outputs=engine_cpu_output,
        )

        engine_gpu_output = gr.Textbox(elem_id="vk_engine_gpu_output", show_label=False)
        engine_gpu_btn = gr.Button("process_engine_gpu", elem_id="vk_engine_gpu_btn")
        engine_gpu_btn.click(
            fn=process_engine_event_bridge_gpu,
            inputs=engine_input,
            outputs=engine_gpu_output,
        )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    demo.launch()
