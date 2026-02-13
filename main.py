# main.py
import io
import json
from fastapi import FastAPI, Request, Response
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage

import gradio as gr

app = FastAPI(title="Virtual MIDI Keyboard - prototype")

# allow local testing from browsers (same host) — adjust in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # in prod lock this down
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


# --- FastAPI route: accepts events JSON and returns .mid bytes ---
@app.post("/save_midi")
async def save_midi(request: Request):
    """
    Expects JSON: {"events": [ {type, note, velocity, time, channel}, ... ]}
    Returns: application/octet-stream or audio/midi bytes of a .mid file
    """
    try:
        payload = await request.json()
    except Exception as e:
        return Response(content=f"Invalid JSON: {e}", status_code=400)

    events = payload.get("events")
    if not isinstance(events, list) or len(events) == 0:
        return Response(content="No events provided", status_code=400)

    mid_bytes = events_to_midbytes(events)
    return Response(content=mid_bytes, media_type="audio/midi")


# --- FastAPI route: serve the standalone keyboard HTML ---
@app.get("/keyboard", response_class=HTMLResponse)
async def keyboard_page():
    with open("keyboard.html", "r", encoding="utf-8") as handle:
        return handle.read()


# --- Gradio UI: embed the keyboard.html inside a simple Blocks app ---
with gr.Blocks() as demo:
    gr.HTML(
        '<iframe src="/keyboard" style="width:100%;height:420px;border:0;"></iframe>'
    )

# Mount Gradio under /app so FastAPI root keeps /save_midi endpoint available
# If you change the path, update the client fetch URL
gr.mount_gradio_app(app, demo, path="/app")

# If you want to run this file directly (uvicorn recommended), nothing else to add.
# Run: uvicorn main:app --reload --port 8000
