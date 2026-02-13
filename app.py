import base64
import html
import io

import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

import gradio as gr


# --- MIDI conversion helper ---
def events_to_midbytes(events, ticks_per_beat=480, tempo_bpm=120):
    """
    events: list of {type:'note_on'|'note_off', note:int, velocity:int, time:float (seconds), channel:int}
    returns: bytes of a .mid file
    """
    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)
    # set tempo meta message
    tempo = mido.bpm2tempo(tempo_bpm)
    track.append(MetaMessage("set_tempo", tempo=tempo, time=0))

    # sort by absolute time
    evs = sorted(events, key=lambda e: e.get("time", 0.0))
    last_time = 0.0
    # convert delta seconds -> delta ticks
    for ev in evs:
        # safety: skip malformed
        if "time" not in ev or "type" not in ev or "note" not in ev:
            continue
        dt_sec = max(0.0, ev["time"] - last_time)
        last_time = ev["time"]
        # ticks = seconds * ticks_per_beat * bpm / 60
        ticks = int(round(dt_sec * (ticks_per_beat * (tempo_bpm)) / 60.0))
        ev_type = ev["type"]
        note = int(ev["note"])
        vel = int(ev.get("velocity", 0))
        channel = int(ev.get("channel", 0))
        if ev_type == "note_on":
            msg = Message(
                "note_on", note=note, velocity=vel, time=ticks, channel=channel
            )
        else:
            # treat anything else as note_off for now
            msg = Message(
                "note_off", note=note, velocity=vel, time=ticks, channel=channel
            )
        track.append(msg)

    # write to bytes
    buf = io.BytesIO()
    mid.save(file=buf)
    buf.seek(0)
    return buf.read()


def save_midi_api(events):
    """
    Gradio API: events is a list of MIDI event dicts from the browser.
    Returns: JSON with base64-encoded MIDI bytes.
    """
    if not isinstance(events, list) or len(events) == 0:
        return {"error": "No events provided"}

    mid_bytes = events_to_midbytes(events)
    midi_b64 = base64.b64encode(mid_bytes).decode("ascii")
    return {"midi_base64": midi_b64}


with open("keyboard.html", "r", encoding="utf-8") as handle:
    keyboard_html = handle.read()

iframe_html = (
    '<iframe srcdoc="'
    + html.escape(keyboard_html, quote=True)
    + '" style="width:100%;height:420px;border:0;"></iframe>'
)

# --- Gradio UI: embed the keyboard HTML in an iframe and expose an API ---
with gr.Blocks() as demo:
    gr.HTML(iframe_html)
    api_input = gr.JSON(visible=False)
    api_output = gr.JSON(visible=False)
    api_trigger = gr.Button(visible=False)
    api_trigger.click(
        save_midi_api,
        inputs=api_input,
        outputs=api_output,
        api_name="save_midi",
    )

# Run locally: python app.py
if __name__ == "__main__":
    demo.launch()
