#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from config import MIDI_DEFAULTS, KEYBOARD_BASE_MIDI, KEYBOARD_OCTAVES


DEFAULT_REPO = "asigalov61/Godzilla-Piano-Transformer"
DEFAULT_FILENAME = (
    "Godzilla_Piano_Chords_Texturing_Transformer_Trained_Model_22708_steps_"
    "0.7515_loss_0.7853_acc.pth"
)

_MODEL_CACHE: dict[str, object] = {}


@dataclass(frozen=True)
class MidiModel:
    model_id: str
    name: str

    def generate_continuation(
        self,
        events: list[dict],
        *,
        tokens: int = 32,
        seed: Optional[int] = None,
    ) -> list[dict]:
        raise NotImplementedError


def ensure_tegridy_tools(base_dir: Path) -> tuple[Path, Path]:
    repo_dir = base_dir / "tegridy-tools"
    tools_dir = repo_dir / "tegridy-tools"
    x_transformer_dir = tools_dir / "X-Transformer"

    if not x_transformer_dir.exists():
        repo_url = "https://github.com/asigalov61/tegridy-tools"
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            subprocess.check_call(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    repo_url,
                    str(repo_dir),
                ]
            )
        except FileNotFoundError as exc:
            raise RuntimeError("git is required to clone tegridy-tools") from exc

    return tools_dir, x_transformer_dir


def add_sys_path(*paths: Path) -> None:
    for path in paths:
        path_str = str(path.resolve())
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def build_model(seq_len: int, pad_idx: int):
    from x_transformer_2_3_1 import AutoregressiveWrapper, Decoder, TransformerWrapper

    model = TransformerWrapper(
        num_tokens=pad_idx + 1,
        max_seq_len=seq_len,
        attn_layers=Decoder(
            dim=2048,
            depth=8,
            heads=32,
            rotary_pos_emb=True,
            attn_flash=True,
        ),
    )
    return AutoregressiveWrapper(model, ignore_index=pad_idx, pad_value=pad_idx)


def resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def load_checkpoint(model, checkpoint_path: Path, device: str) -> None:
    import torch

    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)


def events_to_score_tokens(events: list[dict]) -> list[int]:
    if not events:
        return []

    active: dict[int, float] = {}
    notes: list[tuple[float, float, int]] = []
    sorted_events = sorted(events, key=lambda e: e.get("time", 0.0))

    for event in sorted_events:
        ev_type = event.get("type")
        note = int(event.get("note", 0))
        velocity = int(event.get("velocity", 0))
        time_sec = float(event.get("time", 0.0))

        if ev_type == "note_on" and velocity > 0:
            active[note] = time_sec
        elif ev_type in {"note_off", "note_on"}:
            if note in active:
                start = active.pop(note)
                duration = max(0.0, time_sec - start)
                notes.append((start, duration, note))

    if not notes:
        return []

    notes.sort(key=lambda n: n[0])
    tokens: list[int] = []
    prev_start_ms = 0.0

    for start, duration, pitch in notes:
        start_ms = round(start * 1000.0)
        delta_ms = max(0.0, start_ms - prev_start_ms)
        prev_start_ms = start_ms

        time_tok = max(0, min(127, int(round(delta_ms / 32.0))))
        dur_tok = max(1, min(127, int(round((duration * 1000.0) / 32.0))))
        pitch_tok = max(0, min(127, int(pitch)))

        tokens.extend([time_tok, 128 + dur_tok, 256 + pitch_tok])

    return tokens


def tokens_to_events(
    tokens: Iterable[int],
    *,
    offset_ms: float = 0.0,
    velocity: int | None = None,
) -> list[dict]:
    if velocity is None:
        velocity = MIDI_DEFAULTS["velocity_default"]

    events: list[dict] = []
    time_ms = offset_ms
    duration_ms = 1
    pitch = 60

    for tok in tokens:
        if 0 <= tok < 128:
            time_ms += tok * 32
        elif 128 < tok < 256:
            duration_ms = (tok - 128) * 32
        elif 256 < tok < 384:
            pitch = tok - 256
            on_time = time_ms / 1000.0
            off_time = (time_ms + duration_ms) / 1000.0
            events.append(
                {
                    "type": "note_on",
                    "note": pitch,
                    "velocity": velocity,
                    "time": on_time,
                    "channel": 0,
                }
            )
            events.append(
                {
                    "type": "note_off",
                    "note": pitch,
                    "velocity": 0,
                    "time": off_time,
                    "channel": 0,
                }
            )

    return events


def keyboard_note_range() -> tuple[int, int]:
    min_note = KEYBOARD_BASE_MIDI
    max_note = KEYBOARD_BASE_MIDI + (KEYBOARD_OCTAVES * 12) - 1
    return min_note, max_note


def count_out_of_range_events(events: list[dict]) -> int:
    min_note, max_note = keyboard_note_range()
    return sum(
        1
        for event in events
        if event.get("type") in {"note_on", "note_off"}
        and int(event.get("note", min_note)) not in range(min_note, max_note + 1)
    )


def filter_events_to_keyboard_range(events: list[dict]) -> list[dict]:
    min_note, max_note = keyboard_note_range()
    return [
        event
        for event in events
        if event.get("type") not in {"note_on", "note_off"}
        or min_note <= int(event.get("note", min_note)) <= max_note
    ]


def build_prime_tokens(score_tokens: list[int], seq_len: int) -> list[int]:
    prime = [705, 384, 706]
    if score_tokens:
        max_score = max(0, seq_len - len(prime))
        prime.extend(score_tokens[-max_score:])
    else:
        prime.extend([0, 129, 316])
    return prime


def load_model_cached(
    *,
    repo: str,
    filename: str,
    cache_dir: Path,
    tegridy_dir: Path,
    seq_len: int,
    pad_idx: int,
    device: str,
) -> tuple[object, str, Path]:
    from huggingface_hub import hf_hub_download
    import torch

    cache_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = resolve_device(device)
    cache_key = f"{repo}:{filename}:{seq_len}:{pad_idx}:{resolved_device}"

    if _MODEL_CACHE.get("key") == cache_key:
        return (
            _MODEL_CACHE["model"],
            _MODEL_CACHE["device"],
            _MODEL_CACHE["tools_dir"],
        )

    checkpoint_path = Path(
        hf_hub_download(
            repo_id=repo,
            filename=filename,
            local_dir=str(cache_dir),
            repo_type="model",
        )
    )

    tools_dir, x_transformer_dir = ensure_tegridy_tools(tegridy_dir)
    add_sys_path(x_transformer_dir)

    if resolved_device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model = build_model(seq_len, pad_idx)
    load_checkpoint(model, checkpoint_path, resolved_device)
    model.to(resolved_device)
    model.eval()

    _MODEL_CACHE["key"] = cache_key
    _MODEL_CACHE["model"] = model
    _MODEL_CACHE["device"] = resolved_device
    _MODEL_CACHE["tools_dir"] = tools_dir
    _MODEL_CACHE["checkpoint_path"] = checkpoint_path

    return model, resolved_device, tools_dir


def generate_from_events(
    events: list[dict],
    *,
    generate_tokens: int,
    seed: int | None,
    repo: str,
    filename: str,
    cache_dir: Path,
    tegridy_dir: Path,
    seq_len: int,
    pad_idx: int,
    device: str,
) -> tuple[list[dict], list[int]]:
    import torch

    model, resolved_device, _ = load_model_cached(
        repo=repo,
        filename=filename,
        cache_dir=cache_dir,
        tegridy_dir=tegridy_dir,
        seq_len=seq_len,
        pad_idx=pad_idx,
        device=device,
    )

    if seed is not None:
        torch.manual_seed(seed)
        if resolved_device == "cuda":
            torch.cuda.manual_seed_all(seed)

    score_tokens = events_to_score_tokens(events)
    prime = build_prime_tokens(score_tokens, seq_len)
    prime_tensor = torch.tensor(prime, dtype=torch.long, device=resolved_device)

    out = model.generate(
        prime_tensor,
        generate_tokens,
        return_prime=True,
        eos_token=707,
    )

    tokens = out.detach().cpu().tolist()
    new_tokens = tokens[len(prime) :]

    last_time_ms = 0.0
    if events:
        last_time_ms = max(float(e.get("time", 0.0)) for e in events) * 1000.0

    new_events = tokens_to_events(new_tokens, offset_ms=last_time_ms)
    return new_events, new_tokens


def generate_godzilla_continuation(
    events: list[dict],
    *,
    generate_tokens: int = 32,
    seed: int | None = None,
    device: str = "auto",
) -> tuple[list[dict], list[int]]:
    return generate_from_events(
        events,
        generate_tokens=generate_tokens,
        seed=seed,
        repo=DEFAULT_REPO,
        filename=DEFAULT_FILENAME,
        cache_dir=Path(".cache/godzilla"),
        tegridy_dir=Path("external"),
        seq_len=1536,
        pad_idx=708,
        device=device,
    )


class GodzillaMidiModel(MidiModel):
    def __init__(self) -> None:
        super().__init__(model_id="godzilla", name="Godzilla")

    def generate_continuation(
        self,
        events: list[dict],
        *,
        tokens: int = 32,
        seed: Optional[int] = None,
    ) -> list[dict]:
        new_events, _ = generate_godzilla_continuation(
            events,
            generate_tokens=tokens,
            seed=seed,
            device="auto",
        )
        return new_events


def get_model(model_id: str) -> MidiModel:
    if model_id == "godzilla":
        return GodzillaMidiModel()
    raise ValueError(f"Unknown MIDI model: {model_id}")
