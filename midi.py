"""
MIDI Utilities

Functions for working with MIDI events and files.
"""

import io
import mido
from mido import MidiFile, MidiTrack, Message, MetaMessage
from config import MIDI_DEFAULTS


def events_to_midbytes(events, ticks_per_beat=None, tempo_bpm=None):
    """
    Convert browser MIDI events to a .mid file format.

    Args:
        events: List of dicts with {type, note, velocity, time, channel}
        ticks_per_beat: MIDI ticks per beat resolution (default: from config)
        tempo_bpm: Tempo in beats per minute (default: from config)

    Returns:
        bytes: Complete MIDI file as bytes
    """
    if ticks_per_beat is None:
        ticks_per_beat = MIDI_DEFAULTS["ticks_per_beat"]
    if tempo_bpm is None:
        tempo_bpm = MIDI_DEFAULTS["tempo_bpm"]

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


def validate_event(event: dict) -> bool:
    """
    Validate that an event has required fields.

    Args:
        event: MIDI event dictionary

    Returns:
        bool: True if valid, False otherwise
    """
    required = {"type", "note", "time", "velocity"}
    return all(field in event for field in required)
