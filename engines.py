"""
Virtual MIDI Keyboard - Engines

MIDI processing engines that transform, analyze, or manipulate MIDI events.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


# =============================================================================
# BASE ENGINE CLASS
# =============================================================================


class MIDIEngine(ABC):
    """Abstract base class for MIDI engines"""

    def __init__(self, name: str):
        self.name = name

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


# =============================================================================
# PARROT ENGINE
# =============================================================================


class ParrotEngine(MIDIEngine):
    """
    Parrot Engine - plays back MIDI exactly as recorded.

    This is the simplest engine - it just repeats what the user played.
    """

    def __init__(self):
        super().__init__("Parrot")

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
        }
