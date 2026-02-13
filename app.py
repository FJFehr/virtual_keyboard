"""
Virtual MIDI Keyboard - Gradio Application

This application provides a browser-based MIDI keyboard that can:
- Play notes with various synthesizer sounds (Tone.js)
- Record MIDI events with timestamps
- Export recordings as .mid files
- Support computer keyboard input
- Monitor MIDI events in real-time

File Structure:
- app.py: Gradio server and MIDI conversion (this file)
- keyboard.html: Main UI structure
- static/styles.css: All application styles
- static/keyboard.js: Client-side logic and interactivity
"""

import base64
import html
import io

import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

import gradio as gr


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
with gr.Blocks() as demo:
    gr.HTML(iframe_html)

    # Hidden API endpoint components (required for Gradio 6.x)
    with gr.Row(visible=False):
        api_input = gr.JSON()
        api_output = gr.JSON()
        api_btn = gr.Button("save")

    # Connect API endpoint
    api_btn.click(
        fn=save_midi_api, inputs=api_input, outputs=api_output, api_name="save_midi"
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    demo.launch()
