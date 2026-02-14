"""
Virtual MIDI Keyboard - Engines

This module contains MIDI processing engines that can transform,
analyze, or manipulate MIDI events in various ways.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Callable
import time


# =============================================================================
# BASE ENGINE CLASS
# =============================================================================


class MIDIEngine(ABC):
    """Abstract base class for MIDI engines"""

    def __init__(self, name: str):
        self.name = name
        self.recordings = {}
        self.current_recording_id = None

    @abstractmethod
    def process(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process MIDI events and return transformed events.

        Args:
            events: List of MIDI event dictionaries

        Returns:
            List of processed MIDI event dictionaries
        """
        pass

    def start_recording(self, recording_name: str = None) -> str:
        """Start recording a new performance"""
        if recording_name is None:
            recording_name = f"recording_{int(time.time() * 1000)}"

        self.current_recording_id = recording_name
        self.recordings[recording_name] = {
            "name": recording_name,
            "events": [],
            "created_at": time.time(),
        }
        return recording_name

    def stop_recording(self) -> Dict[str, Any]:
        """Stop recording and return the recording"""
        if not self.current_recording_id:
            return None

        recording_id = self.current_recording_id
        self.current_recording_id = None
        return self.recordings[recording_id]

    def add_event(self, event: Dict[str, Any]) -> bool:
        """Add event to current recording"""
        if not self.current_recording_id:
            return False

        recording = self.recordings[self.current_recording_id]
        recording["events"].append(event)
        return True

    def get_recording(self, recording_id: str) -> Dict[str, Any]:
        """Get a recording by ID"""
        return self.recordings.get(recording_id)

    def get_all_recordings(self) -> List[Dict[str, Any]]:
        """Get all recordings"""
        return list(self.recordings.values())

    def delete_recording(self, recording_id: str) -> bool:
        """Delete a recording"""
        if recording_id in self.recordings:
            if self.current_recording_id == recording_id:
                self.stop_recording()
            del self.recordings[recording_id]
            return True
        return False


# =============================================================================
# PARROT ENGINE - Returns exactly what was played
# =============================================================================


class ParrotEngine(MIDIEngine):
    """
    Parrot Engine - Captures and plays back MIDI exactly as recorded.

    This is the simplest engine - it just repeats what the user played.
    Perfect for learning and basic playback.
    """

    def __init__(self):
        super().__init__("Parrot")
        self.description = "Captures and plays back MIDI exactly as recorded"

    def process(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process MIDI events - in Parrot mode, just return them unchanged.

        Args:
            events: List of MIDI event dictionaries

        Returns:
            The same events unmodified
        """
        if not events:
            return []

        # Simply return the events as-is
        processed_events = []
        for event in events:
            processed_events.append(
                {
                    "type": event.get("type"),
                    "note": event.get("note"),
                    "velocity": event.get("velocity"),
                    "time": event.get("time"),
                    "channel": event.get("channel", 0),
                }
            )

        return processed_events


# =============================================================================
# ENGINE REGISTRY
# =============================================================================


class EngineRegistry:
    """Registry for managing available MIDI engines"""

    _engines = {"parrot": ParrotEngine}

    @classmethod
    def register(cls, engine_id: str, engine_class: type):
        """Register a new engine"""
        cls._engines[engine_id] = engine_class

    @classmethod
    def get_engine(cls, engine_id: str) -> MIDIEngine:
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
            "description": getattr(engine, "description", "No description"),
        }


# =============================================================================
# GLOBAL ENGINE INSTANCES
# =============================================================================

# Create default engine instance
default_engine = ParrotEngine()
