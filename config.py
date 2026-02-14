"""
Virtual MIDI Keyboard Configuration

Centralized configuration for instruments, keyboard layout, and defaults.
This is the single source of truth for all settings.
"""

# =============================================================================
# KEYBOARD LAYOUT
# =============================================================================

KEYBOARD_BASE_MIDI = 60  # C4
KEYBOARD_OCTAVES = 2
KEYBOARD_POLYPHONY = 24

# Keyboard key layout (white keys first, then black keys in order)
KEYBOARD_KEYS = [
    {"midi": 60, "name": "C4", "type": "white"},
    {"midi": 61, "name": "C#4", "type": "black"},
    {"midi": 62, "name": "D4", "type": "white"},
    {"midi": 63, "name": "D#4", "type": "black"},
    {"midi": 64, "name": "E4", "type": "white"},
    {"midi": 65, "name": "F4", "type": "white"},
    {"midi": 66, "name": "F#4", "type": "black"},
    {"midi": 67, "name": "G4", "type": "white"},
    {"midi": 68, "name": "G#4", "type": "black"},
    {"midi": 69, "name": "A4", "type": "white"},
    {"midi": 70, "name": "A#4", "type": "black"},
    {"midi": 71, "name": "B4", "type": "white"},
    {"midi": 72, "name": "C5", "type": "white"},
    {"midi": 73, "name": "C#5", "type": "black"},
    {"midi": 74, "name": "D5", "type": "white"},
    {"midi": 75, "name": "D#5", "type": "black"},
    {"midi": 76, "name": "E5", "type": "white"},
    {"midi": 77, "name": "F5", "type": "white"},
    {"midi": 78, "name": "F#5", "type": "black"},
    {"midi": 79, "name": "G5", "type": "white"},
    {"midi": 80, "name": "G#5", "type": "black"},
    {"midi": 81, "name": "A5", "type": "white"},
    {"midi": 82, "name": "A#5", "type": "black"},
    {"midi": 83, "name": "B5", "type": "white"},
]

# Computer keyboard shortcuts to MIDI notes
KEYBOARD_SHORTCUTS = {
    60: "A",  # C4
    61: "W",  # C#4
    62: "S",  # D4
    63: "E",  # D#4
    64: "D",  # E4
    65: "F",  # F4
    66: "T",  # F#4
    67: "G",  # G4
    68: "Y",  # G#4
    69: "H",  # A4
    70: "U",  # A#4
    71: "J",  # B4
}

# =============================================================================
# MIDI DEFAULTS
# =============================================================================

MIDI_DEFAULTS = {
    "tempo_bpm": 120,
    "ticks_per_beat": 480,
    "velocity_default": 100,
}

# =============================================================================
# INSTRUMENTS
# =============================================================================

INSTRUMENTS = {
    "synth": {
        "name": "Synth",
        "type": "Synth",
        "oscillator": "sine",
        "envelope": {
            "attack": 0.005,
            "decay": 0.1,
            "sustain": 0.3,
            "release": 1,
        },
    },
    "piano": {
        "name": "Piano",
        "type": "Synth",
        "oscillator": "triangle",
        "envelope": {
            "attack": 0.001,
            "decay": 0.2,
            "sustain": 0.1,
            "release": 2,
        },
    },
    "organ": {
        "name": "Organ",
        "type": "Synth",
        "oscillator": "sine4",
        "envelope": {
            "attack": 0.001,
            "decay": 0.0,
            "sustain": 1.0,
            "release": 0.1,
        },
    },
    "bass": {
        "name": "Bass",
        "type": "Synth",
        "oscillator": "sawtooth",
        "envelope": {
            "attack": 0.01,
            "decay": 0.1,
            "sustain": 0.4,
            "release": 1.5,
        },
    },
    "pluck": {
        "name": "Pluck",
        "type": "Synth",
        "oscillator": "triangle",
        "envelope": {
            "attack": 0.001,
            "decay": 0.3,
            "sustain": 0.0,
            "release": 0.3,
        },
    },
    "fm": {
        "name": "FM Synth",
        "type": "FMSynth",
        "harmonicity": 3,
        "modulationIndex": 10,
        "envelope": {
            "attack": 0.01,
            "decay": 0.2,
            "sustain": 0.2,
            "release": 1,
        },
    },
}
