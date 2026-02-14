"""
Virtual MIDI Keyboard - Engines

MIDI processing engines that transform, analyze, or manipulate MIDI events.
"""

from typing import List, Dict, Any

from midi_model import (
    count_out_of_range_events,
    filter_events_to_keyboard_range,
    get_model,
)


# =============================================================================
# PARROT ENGINE
# =============================================================================


class ParrotEngine:
    """
    Parrot Engine - plays back MIDI exactly as recorded.

    This is the simplest engine - it just repeats what the user played.
    """

    def __init__(self):
        self.name = "Parrot"

    def process(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return events unchanged"""
        if not events:
            return []
        return [
            {
                "type": e.get("type"),
                "note": e.get("note"),
                "velocity": e.get("velocity"),
                "time": e.get("time"),
                "channel": e.get("channel", 0),
            }
            for e in events
        ]


# =============================================================================
# REVERSE PARROT ENGINE
# =============================================================================


class ReverseParrotEngine:
    """
    Reverse Parrot Engine - plays back MIDI in reverse order.

    Takes the recorded performance and reverses the sequence of notes,
    playing them backwards while maintaining their timing relationships.
    """

    def __init__(self):
        self.name = "Reverse Parrot"

    def process(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Reverse the sequence of note numbers while keeping timing and event types"""
        if not events:
            return []

        # Separate note_on and note_off events
        note_on_events = [e for e in events if e.get("type") == "note_on"]
        note_off_events = [e for e in events if e.get("type") == "note_off"]

        # Extract note numbers from note_on events and reverse them
        on_notes = [e.get("note") for e in note_on_events]
        reversed_on_notes = list(reversed(on_notes))

        # Extract note numbers from note_off events and reverse them
        off_notes = [e.get("note") for e in note_off_events]
        reversed_off_notes = list(reversed(off_notes))

        # Reconstruct events with reversed notes but original structure
        result = []
        on_index = 0
        off_index = 0

        for event in events:
            if event.get("type") == "note_on":
                result.append(
                    {
                        "type": "note_on",
                        "note": reversed_on_notes[on_index],
                        "velocity": event.get("velocity"),
                        "time": event.get("time"),
                        "channel": event.get("channel", 0),
                    }
                )
                on_index += 1
            elif event.get("type") == "note_off":
                result.append(
                    {
                        "type": "note_off",
                        "note": reversed_off_notes[off_index],
                        "velocity": event.get("velocity"),
                        "time": event.get("time"),
                        "channel": event.get("channel", 0),
                    }
                )
                off_index += 1

        return result


# =============================================================================
# GODZILLA CONTINUATION ENGINE
# =============================================================================


class GodzillaContinuationEngine:
    """
    Continue a short MIDI phrase with the Godzilla Piano Transformer.

    Generates a small continuation and appends it after the input events.
    """

    def __init__(self, generate_tokens: int = 32):
        self.name = "Godzilla"
        self.generate_tokens = generate_tokens

    def process(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not events:
            return []

        model = get_model("godzilla")
        new_events = model.generate_continuation(
            events,
            tokens=self.generate_tokens,
            seed=None,
        )
        out_of_range = count_out_of_range_events(new_events)
        if out_of_range:
            print(f"Godzilla: dropped {out_of_range} out-of-range events")
        return filter_events_to_keyboard_range(new_events)


# =============================================================================
# ENGINE REGISTRY
# =============================================================================


class EngineRegistry:
    """Registry for managing available MIDI engines"""

    _engines = {
        "parrot": ParrotEngine,
        "reverse_parrot": ReverseParrotEngine,
        "godzilla_continue": GodzillaContinuationEngine,
    }

    @classmethod
    def register(cls, engine_id: str, engine_class: type):
        """Register a new engine"""
        cls._engines[engine_id] = engine_class

    @classmethod
    def get_engine(cls, engine_id: str):
        """Get an engine instance by ID"""
        if engine_id not in cls._engines:
            raise ValueError(f"Unknown engine: {engine_id}")
        return cls._engines[engine_id]()

    @classmethod
    def list_engines(cls) -> List[str]:
        """List all available engines"""
        return list(cls._engines.keys())

    @classmethod
    def get_engine_info(cls, engine_id: str) -> Dict[str, str]:
        """Get info about an engine"""
        if engine_id not in cls._engines:
            raise ValueError(f"Unknown engine: {engine_id}")
        engine = cls._engines[engine_id]()
        return {
            "id": engine_id,
            "name": engine.name,
        }
