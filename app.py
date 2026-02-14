"""
Virtual MIDI Keyboard - Gradio Application

This application provides a browser-based MIDI keyboard that can:
- Play notes with various synthesizer sounds (Tone.js)
- Record MIDI events with timestamps
- Process recordings through various engines
- Export recordings as .mid files
- Support computer keyboard input
- Monitor MIDI events in real-time

File Structure:
- app.py: Gradio server and MIDI conversion (this file)
- engines.py: MIDI processing engines
- keyboard.html: Main UI structure
- static/styles.css: All application styles
- static/keyboard.js: Client-side logic and interactivity
- static/engine.js: Client-side engine abstraction
"""

import base64
import html
import io
import json

import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

import gradio as gr
from engines import EngineRegistry, ParrotEngine


# =============================================================================
# MIDI CONVERSION
# =============================================================================


def events_to_midbytes(events, ticks_per_beat=480, tempo_bpm=120):
    """
    Convert browser MIDI events to a .mid file format.

    Args:
        events: List of dicts with {type, note, velocity, time, channel}
        ticks_per_beat: MIDI ticks per beat resolution (default: 480)
        tempo_bpm: Tempo in beats per minute (default: 120)

    Returns:
        bytes: Complete MIDI file as bytes
    """
    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)

    # Set tempo meta message
    tempo = mido.bpm2tempo(tempo_bpm)
    track.append(MetaMessage("set_tempo", tempo=tempo, time=0))

    # Sort events by time and convert to MIDI messages
    evs = sorted(events, key=lambda e: e.get("time", 0.0))
    last_time = 0.0
    for ev in evs:
        # Skip malformed events
        if "time" not in ev or "type" not in ev or "note" not in ev:
            continue

        # Calculate delta time in ticks
        dt_sec = max(0.0, ev["time"] - last_time)
        last_time = ev["time"]
        ticks = int(round(dt_sec * (ticks_per_beat * tempo_bpm) / 60.0))
        # Create MIDI message
        ev_type = ev["type"]
        note = int(ev["note"])
        vel = int(ev.get("velocity", 0))
        channel = int(ev.get("channel", 0))

        if ev_type == "note_on":
            msg = Message(
                "note_on", note=note, velocity=vel, time=ticks, channel=channel
            )
        else:
            msg = Message(
                "note_off", note=note, velocity=vel, time=ticks, channel=channel
            )

        track.append(msg)

    # Write to bytes buffer
    buf = io.BytesIO()
    mid.save(file=buf)
    buf.seek(0)
    return buf.read()


# =============================================================================
# API ENDPOINT
# =============================================================================


def save_midi_api(events):
    """
    Gradio API endpoint for converting recorded events to MIDI file.

    Args:
        events: List of MIDI event dictionaries from the browser

    Returns:
        Dict with 'midi_base64' or 'error' key
    """
    if not isinstance(events, list) or len(events) == 0:
        return {"error": "No events provided"}

    mid_bytes = events_to_midbytes(events)
    midi_b64 = base64.b64encode(mid_bytes).decode("ascii")
    return {"midi_base64": midi_b64}


# =============================================================================
# ENGINE API ENDPOINTS
# =============================================================================


def list_engines():
    """
    API endpoint to list available MIDI engines

    Returns:
        List of engine info dictionaries
    """
    engines = EngineRegistry.list_engines()
    return {"engines": [EngineRegistry.get_engine_info(e) for e in engines]}


def process_with_engine(engine_id: str, events: list):
    """
    API endpoint to process MIDI events through an engine

    Args:
        engine_id: The engine to use (e.g. 'parrot')
        events: List of MIDI event dictionaries

    Returns:
        Processed events or error message
    """
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


def process_engine_api(payload: dict):
    """
    Wrapper API endpoint that accepts JSON payload

    Args:
        payload: Dict with 'engine_id' and 'events' keys

    Returns:
        Processed events or error message
    """
    try:
        print(f"[DEBUG] process_engine_api called with payload type: {type(payload)}")
        print(
            f"[DEBUG] payload keys: {payload.keys() if isinstance(payload, dict) else 'N/A'}"
        )

        # Handle both direct dict and wrapped dict formats
        data = payload
        if isinstance(payload, dict) and "data" in payload:
            # If wrapped in a data field, unwrap it
            data = payload["data"]
            if isinstance(data, list) and len(data) > 0:
                data = data[0]

        print(
            f"[DEBUG] extracted data type: {type(data)}, has engine_id: {'engine_id' in data if isinstance(data, dict) else False}"
        )

        engine_id = data.get("engine_id") if isinstance(data, dict) else None
        events = data.get("events", []) if isinstance(data, dict) else []

        print(
            f"[DEBUG] engine_id: {engine_id}, events count: {len(events) if isinstance(events, list) else 'N/A'}"
        )

        result = process_with_engine(engine_id, events)
        print(
            f"[DEBUG] process_engine_api returning: {result.keys() if isinstance(result, dict) else type(result)}"
        )
        return result
    except Exception as e:
        print(f"[DEBUG] Exception in process_engine_api: {str(e)}")
        import traceback

        traceback.print_exc()
        return {"error": f"API error: {str(e)}"}


# =============================================================================
# GRADIO APPLICATION
# =============================================================================

# Load and combine HTML, CSS, and JS for the iframe
with open("keyboard.html", "r", encoding="utf-8") as f:
    html_content = f.read()

with open("static/styles.css", "r", encoding="utf-8") as f:
    css_content = f.read()

with open("static/keyboard.js", "r", encoding="utf-8") as f:
    js_content = f.read()

# Inject CSS and JS into the HTML
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

# Create Gradio interface
with gr.Blocks(title="Virtual MIDI Keyboard") as demo:
    gr.HTML(iframe_html)

    # Hidden API endpoints using Gradio's function API
    with gr.Group(visible=False) as api_group:
        # Process engine endpoint
        engine_input = gr.Textbox(label="engine_payload")
        engine_output = gr.Textbox(label="engine_result")

        def call_engine_api(payload_json):
            """Wrapper to call engine API with JSON input"""
            import json

            try:
                payload = (
                    json.loads(payload_json)
                    if isinstance(payload_json, str)
                    else payload_json
                )
                result = process_engine_api(payload)
                return json.dumps(result)
            except Exception as e:
                return json.dumps({"error": str(e)})

        engine_btn = gr.Button("process_engine", visible=False)
        engine_btn.click(
            fn=call_engine_api,
            inputs=engine_input,
            outputs=engine_output,
            api_name="process_engine",
        )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    demo.launch()
