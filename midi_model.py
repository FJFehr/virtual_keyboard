#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import inspect
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

from config import MIDI_DEFAULTS, KEYBOARD_BASE_MIDI, KEYBOARD_OCTAVES

DEFAULT_REPO = "asigalov61/Godzilla-Piano-Transformer"
DEFAULT_FILENAME = (
    "Godzilla_Piano_Transformer_No_Velocity_Trained_Model_21113_steps_"
    "0.3454_loss_0.895_acc.pth"
)
KNOWN_GODZILLA_CHECKPOINTS = [
    "Godzilla_Piano_Transformer_No_Velocity_Trained_Model_21113_steps_0.3454_loss_0.895_acc.pth",
    "Godzilla_Piano_Transformer_No_Velocity_Trained_Model_14903_steps_0.4874_loss_0.8571_acc.pth",
    "Godzilla_Piano_Transformer_No_Velocity_Trained_Model_32503_steps_0.6553_loss_0.8065_acc.pth",
    "Godzilla_Piano_Chords_Texturing_Transformer_Trained_Model_22708_steps_0.7515_loss_0.7853_acc.pth",
]

_MODEL_CACHE: dict[str, object] = {}
PROMPT_MAX_SECONDS = 18.0
PROMPT_PHRASE_GAP_SEC = 0.6
PROMPT_TARGET_CENTER_MIDI = 67
PROMPT_MAX_TRANSPOSE = 12
DEFAULT_MODEL_DIM = 2048
DEFAULT_MODEL_DEPTH = 8
DEFAULT_MODEL_HEADS = 32
DEFAULT_SEQ_LEN = 1536
DEFAULT_PAD_IDX = 708


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
        temperature: float = 0.9,
        top_p: float = 0.95,
        num_candidates: int = 1,
        request: Any | None = None,
        device: str = "auto",
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


def preload_godzilla_assets(
    *,
    repo: str = DEFAULT_REPO,
    filename: str = DEFAULT_FILENAME,
    cache_dir: Path = Path(".cache/godzilla"),
    tegridy_dir: Path = Path("external"),
) -> Path:
    """
    Download model checkpoint and Tegridy tools during app startup.
    """
    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = download_checkpoint_with_fallback(
        repo=repo,
        filename=filename,
        cache_dir=cache_dir,
    )
    ensure_tegridy_tools(tegridy_dir)
    return checkpoint_path


def preload_godzilla_model(
    *,
    repo: str = DEFAULT_REPO,
    filename: str = DEFAULT_FILENAME,
    cache_dir: Path = Path(".cache/godzilla"),
    tegridy_dir: Path = Path("external"),
    seq_len: int = DEFAULT_SEQ_LEN,
    pad_idx: int = DEFAULT_PAD_IDX,
    device: str = "cpu",
) -> dict[str, Any]:
    model, resolved_device, _, resolved_seq_len, resolved_pad_idx = load_model_cached(
        repo=repo,
        filename=filename,
        cache_dir=cache_dir,
        tegridy_dir=tegridy_dir,
        seq_len=seq_len,
        pad_idx=pad_idx,
        device=device,
    )
    return {
        "model_loaded": model is not None,
        "device": resolved_device,
        "seq_len": resolved_seq_len,
        "pad_idx": resolved_pad_idx,
    }


def candidate_checkpoint_filenames(primary: str) -> list[str]:
    ordered = [primary]
    for checkpoint in KNOWN_GODZILLA_CHECKPOINTS:
        if checkpoint not in ordered:
            ordered.append(checkpoint)
    return ordered


def download_checkpoint_with_fallback(
    *,
    repo: str,
    filename: str,
    cache_dir: Path,
) -> Path:
    from huggingface_hub import hf_hub_download

    cache_dir.mkdir(parents=True, exist_ok=True)
    attempted: list[str] = []
    last_error: Exception | None = None

    for candidate in candidate_checkpoint_filenames(filename):
        try:
            return Path(
                hf_hub_download(
                    repo_id=repo,
                    filename=candidate,
                    local_dir=str(cache_dir),
                    repo_type="model",
                )
            )
        except Exception as exc:
            attempted.append(candidate)
            last_error = exc
            message = str(exc)
            if "Entry Not Found" in message or "404" in message:
                continue
            raise

    raise RuntimeError(
        f"Could not download any Godzilla checkpoint. Tried: {attempted}. "
        f"Last error: {last_error}"
    )


def add_sys_path(*paths: Path) -> None:
    for path in paths:
        path_str = str(path.resolve())
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def build_model(
    seq_len: int,
    pad_idx: int,
    *,
    dim: int = DEFAULT_MODEL_DIM,
    depth: int = DEFAULT_MODEL_DEPTH,
    heads: int = DEFAULT_MODEL_HEADS,
):
    from x_transformer_2_3_1 import AutoregressiveWrapper, Decoder, TransformerWrapper

    model = TransformerWrapper(
        num_tokens=pad_idx + 1,
        max_seq_len=seq_len,
        attn_layers=Decoder(
            dim=dim,
            depth=depth,
            heads=heads,
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


def load_checkpoint_state(checkpoint_path: Path, device: str) -> dict[str, Any]:
    import torch

    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise RuntimeError(f"Unexpected checkpoint format at {checkpoint_path}")
    return state


def infer_layer_entries_from_state(state: dict[str, Any]) -> int:
    layer_ids: set[int] = set()
    pattern = re.compile(r"^net\.attn_layers\.layers\.(\d+)\.")
    for key in state.keys():
        match = pattern.match(key)
        if match:
            layer_ids.add(int(match.group(1)))
    if not layer_ids:
        return DEFAULT_MODEL_DEPTH * 2
    return max(layer_ids) + 1


def infer_model_shape_from_state(
    state: dict[str, Any],
    *,
    fallback_seq_len: int,
    fallback_pad_idx: int,
) -> tuple[int, int, int, list[int]]:
    emb = state.get("net.token_emb.emb.weight")
    if emb is not None and hasattr(emb, "shape") and len(emb.shape) == 2:
        num_tokens = int(emb.shape[0])
        dim = int(emb.shape[1])
    else:
        num_tokens = fallback_pad_idx + 1
        dim = DEFAULT_MODEL_DIM

    pad_idx = max(0, num_tokens - 1)
    seq_len = 4096 if num_tokens <= 385 else fallback_seq_len

    layer_entries = infer_layer_entries_from_state(state)
    depth_candidates: list[int] = []
    for candidate in [max(1, layer_entries // 2), layer_entries]:
        if candidate not in depth_candidates:
            depth_candidates.append(candidate)

    return seq_len, pad_idx, dim, depth_candidates


def adapt_state_for_model(
    raw_state: dict[str, Any],
    model_state: dict[str, Any],
) -> dict[str, Any]:
    adapted: dict[str, Any] = {}
    model_keys = set(model_state.keys())

    for key, value in raw_state.items():
        if key in model_keys:
            adapted[key] = value
            continue
        if key.endswith(".weight"):
            gamma_key = key[:-7] + ".gamma"
            if gamma_key in model_keys:
                adapted[gamma_key] = value
                continue
        if key.endswith(".gamma"):
            weight_key = key[:-6] + ".weight"
            if weight_key in model_keys:
                adapted[weight_key] = value
                continue

    # Fill optional affine params when norm conventions differ.
    for key, tensor in model_state.items():
        if key in adapted:
            continue
        if key.endswith(".bias"):
            adapted[key] = tensor.new_zeros(tensor.shape)
        elif key.endswith(".gamma"):
            adapted[key] = tensor.new_ones(tensor.shape)

    return adapted


def sanitize_events(events: list[dict]) -> list[dict]:
    cleaned: list[dict] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        ev_type = event.get("type")
        if ev_type not in {"note_on", "note_off"}:
            continue
        note = max(0, min(127, int(event.get("note", 0))))
        velocity = max(0, min(127, int(event.get("velocity", 0))))
        time_sec = max(0.0, float(event.get("time", 0.0)))
        cleaned.append(
            {
                "type": ev_type,
                "note": note,
                "velocity": velocity,
                "time": time_sec,
                "channel": int(event.get("channel", 0)),
            }
        )
    cleaned.sort(key=lambda e: float(e.get("time", 0.0)))
    return cleaned


def extract_prompt_window(
    events: list[dict],
    *,
    max_seconds: float = PROMPT_MAX_SECONDS,
    phrase_gap_sec: float = PROMPT_PHRASE_GAP_SEC,
) -> list[dict]:
    cleaned = sanitize_events(events)
    if not cleaned:
        return []

    last_time = max(float(e.get("time", 0.0)) for e in cleaned)
    recent_cut = max(0.0, last_time - max_seconds)

    note_on_times = [
        float(e.get("time", 0.0))
        for e in cleaned
        if e.get("type") == "note_on" and int(e.get("velocity", 0)) > 0
    ]
    if len(note_on_times) < 2:
        return [e for e in cleaned if float(e.get("time", 0.0)) >= recent_cut]

    phrase_starts = [note_on_times[0]]
    for i in range(1, len(note_on_times)):
        if note_on_times[i] - note_on_times[i - 1] >= phrase_gap_sec:
            phrase_starts.append(note_on_times[i])

    # Keep the last 1-2 phrases for coherent continuation.
    phrase_cut = phrase_starts[-2] if len(phrase_starts) >= 2 else phrase_starts[-1]
    cut = max(recent_cut, phrase_cut)
    return [e for e in cleaned if float(e.get("time", 0.0)) >= cut]


def estimate_input_velocity(events: list[dict], default: int = 100) -> int:
    velocities = [
        int(e.get("velocity", 0))
        for e in events
        if e.get("type") == "note_on" and int(e.get("velocity", 0)) > 0
    ]
    if not velocities:
        return default
    avg = round(sum(velocities) / len(velocities))
    return max(40, min(120, avg))


def compute_transpose_shift(
    events: list[dict],
    *,
    target_center_midi: int = PROMPT_TARGET_CENTER_MIDI,
    max_transpose: int = PROMPT_MAX_TRANSPOSE,
) -> int:
    pitches = [
        int(e.get("note", 0))
        for e in events
        if e.get("type") == "note_on" and int(e.get("velocity", 0)) > 0
    ]
    if not pitches:
        return 0
    pitches.sort()
    median_pitch = pitches[len(pitches) // 2]
    shift = int(round(target_center_midi - median_pitch))
    return max(-max_transpose, min(max_transpose, shift))


def transpose_events(events: list[dict], semitones: int) -> list[dict]:
    if semitones == 0:
        return [dict(event) for event in events]
    out: list[dict] = []
    for event in events:
        copied = dict(event)
        if copied.get("type") in {"note_on", "note_off"}:
            copied["note"] = max(0, min(127, int(copied.get("note", 0)) + semitones))
        out.append(copied)
    return out


def events_to_score_tokens(events: list[dict]) -> list[int]:
    if not events:
        return []

    active: dict[int, list[tuple[float, int]]] = {}
    notes: list[tuple[float, float, int]] = []
    sorted_events = sorted(events, key=lambda e: e.get("time", 0.0))

    for event in sorted_events:
        ev_type = event.get("type")
        note = int(event.get("note", 0))
        velocity = int(event.get("velocity", 0))
        time_sec = float(event.get("time", 0.0))

        if ev_type == "note_on" and velocity > 0:
            active.setdefault(note, []).append((time_sec, velocity))
        elif ev_type in {"note_off", "note_on"}:
            if note in active and active[note]:
                start, _ = active[note].pop(0)
                duration = max(0.05, time_sec - start)
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


def fold_note_to_keyboard_range(note: int) -> int:
    min_note, max_note = keyboard_note_range()
    folded = int(note)
    while folded < min_note:
        folded += 12
    while folded > max_note:
        folded -= 12
    return max(min_note, min(max_note, folded))


def fold_events_to_keyboard_range(events: list[dict]) -> list[dict]:
    out: list[dict] = []
    for event in events:
        copied = dict(event)
        if copied.get("type") in {"note_on", "note_off"}:
            copied["note"] = fold_note_to_keyboard_range(int(copied.get("note", 0)))
        out.append(copied)
    return out


def resolve_eos_token(pad_idx: int) -> int:
    # Legacy Godzilla checkpoints use 707 as EOS with 708 pad.
    if pad_idx >= 708:
        return 707
    return pad_idx


def build_prime_tokens(score_tokens: list[int], seq_len: int, pad_idx: int) -> list[int]:
    num_tokens = pad_idx + 1
    if pad_idx >= 708:
        prime = [705, 384, 706]
        if score_tokens:
            max_score = max(0, seq_len - len(prime))
            prime.extend(score_tokens[-max_score:])
        else:
            prime.extend([0, 129, 316])
    else:
        if score_tokens:
            prime = score_tokens[-max(1, seq_len) :]
        else:
            prime = [0, 129, 316]

    # Keep prime tokens valid for current vocabulary.
    return [max(0, min(num_tokens - 1, int(tok))) for tok in prime]


def build_generate_kwargs(
    model,
    temperature: float,
    top_p: float,
    eos_token: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "return_prime": True,
        "eos_token": eos_token,
    }
    try:
        signature = inspect.signature(model.generate)
    except (TypeError, ValueError):
        return kwargs

    params = signature.parameters
    safe_temperature = max(0.2, min(1.5, float(temperature)))
    safe_top_p = max(0.5, min(0.99, float(top_p)))

    if "temperature" in params:
        kwargs["temperature"] = safe_temperature
    if "top_p" in params:
        kwargs["top_p"] = safe_top_p
    elif "filter_thres" in params:
        kwargs["filter_thres"] = safe_top_p
    return kwargs


def generate_tokens_sample(
    model,
    prime_tensor,
    generate_tokens: int,
    *,
    temperature: float,
    top_p: float,
    eos_token: int,
) -> list[int]:
    kwargs = build_generate_kwargs(model, temperature, top_p, eos_token)
    try:
        out = model.generate(
            prime_tensor,
            generate_tokens,
            **kwargs,
        )
    except TypeError:
        out = model.generate(
            prime_tensor,
            generate_tokens,
            return_prime=True,
            eos_token=eos_token,
        )
    return out.detach().cpu().tolist()


def score_candidate_events(events: list[dict], prompt_events: list[dict]) -> float:
    notes = [
        int(e.get("note", 0))
        for e in events
        if e.get("type") == "note_on" and int(e.get("velocity", 0)) > 0
    ]
    if not notes:
        return -1e6

    prompt_notes = [
        int(e.get("note", 0))
        for e in prompt_events
        if e.get("type") == "note_on" and int(e.get("velocity", 0)) > 0
    ]

    min_note, max_note = keyboard_note_range()
    out_of_range = sum(1 for note in notes if note < min_note or note > max_note)
    repeats = sum(1 for i in range(1, len(notes)) if notes[i] == notes[i - 1])
    big_leaps = sum(max(0, abs(notes[i] - notes[i - 1]) - 7) for i in range(1, len(notes)))

    score = 0.0
    score += min(len(notes), 24) * 0.25
    score -= out_of_range * 3.5
    score -= repeats * 0.2
    score -= big_leaps * 0.08

    if prompt_notes:
        prompt_center = sum(prompt_notes) / len(prompt_notes)
        notes_center = sum(notes) / len(notes)
        score -= abs(notes_center - prompt_center) * 0.04
        score -= abs(notes[0] - prompt_notes[-1]) * 0.03

    return score


def load_model_cached(
    *,
    repo: str,
    filename: str,
    cache_dir: Path,
    tegridy_dir: Path,
    seq_len: int,
    pad_idx: int,
    device: str,
) -> tuple[object, str, Path, int, int]:
    import torch

    cache_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = resolve_device(device)

    checkpoint_path = download_checkpoint_with_fallback(
        repo=repo,
        filename=filename,
        cache_dir=cache_dir,
    )

    tools_dir, x_transformer_dir = ensure_tegridy_tools(tegridy_dir)
    add_sys_path(x_transformer_dir)

    if resolved_device == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    raw_state = load_checkpoint_state(checkpoint_path, "cpu")
    inferred_seq_len, inferred_pad_idx, inferred_dim, depth_candidates = infer_model_shape_from_state(
        raw_state,
        fallback_seq_len=seq_len,
        fallback_pad_idx=pad_idx,
    )

    cache_key = (
        f"{repo}:{filename}:{resolved_device}:{inferred_seq_len}:{inferred_pad_idx}:"
        f"{inferred_dim}:{'-'.join(str(d) for d in depth_candidates)}"
    )
    if _MODEL_CACHE.get("key") == cache_key:
        return (
            _MODEL_CACHE["model"],
            _MODEL_CACHE["device"],
            _MODEL_CACHE["tools_dir"],
            _MODEL_CACHE["seq_len"],
            _MODEL_CACHE["pad_idx"],
        )

    selected_model = None
    selected_depth = None
    last_error: RuntimeError | None = None
    for depth in depth_candidates:
        try:
            model = build_model(
                inferred_seq_len,
                inferred_pad_idx,
                dim=inferred_dim,
                depth=depth,
                heads=DEFAULT_MODEL_HEADS,
            )
            adapted_state = adapt_state_for_model(raw_state, model.state_dict())
            missing, _ = model.load_state_dict(adapted_state, strict=False)
            critical_missing = [
                key
                for key in missing
                if not (key.endswith(".bias") or key.endswith(".gamma"))
            ]
            if critical_missing:
                raise RuntimeError(
                    f"critical missing keys for depth={depth}: {critical_missing[:5]}"
                )
            selected_model = model
            selected_depth = depth
            break
        except RuntimeError as exc:
            last_error = exc

    if selected_model is None:
        raise RuntimeError(
            f"Could not load checkpoint {filename} with inferred configs. "
            f"Tried depths={depth_candidates}. Last error: {last_error}"
        )

    model = selected_model
    model.to(resolved_device)
    model.eval()
    if resolved_device == "cuda":
        first_param = next(model.parameters(), None)
        if first_param is None or not first_param.is_cuda:
            raise RuntimeError("Godzilla model failed to move to CUDA device")

    print(
        "Loaded Godzilla checkpoint config:",
        {
            "filename": filename,
            "seq_len": inferred_seq_len,
            "pad_idx": inferred_pad_idx,
            "dim": inferred_dim,
            "depth": selected_depth,
            "device": resolved_device,
            "cuda_available": torch.cuda.is_available(),
        },
    )

    _MODEL_CACHE["key"] = cache_key
    _MODEL_CACHE["model"] = model
    _MODEL_CACHE["device"] = resolved_device
    _MODEL_CACHE["tools_dir"] = tools_dir
    _MODEL_CACHE["checkpoint_path"] = checkpoint_path
    _MODEL_CACHE["seq_len"] = inferred_seq_len
    _MODEL_CACHE["pad_idx"] = inferred_pad_idx

    return model, resolved_device, tools_dir, inferred_seq_len, inferred_pad_idx


def generate_from_events(
    events: list[dict],
    *,
    generate_tokens: int,
    seed: int | None,
    temperature: float,
    top_p: float,
    num_candidates: int,
    repo: str,
    filename: str,
    cache_dir: Path,
    tegridy_dir: Path,
    seq_len: int,
    pad_idx: int,
    device: str,
) -> tuple[list[dict], list[int]]:
    import torch

    model, resolved_device, _, resolved_seq_len, resolved_pad_idx = load_model_cached(
        repo=repo,
        filename=filename,
        cache_dir=cache_dir,
        tegridy_dir=tegridy_dir,
        seq_len=seq_len,
        pad_idx=pad_idx,
        device=device,
    )

    prompt_events = extract_prompt_window(events)
    transpose_shift = compute_transpose_shift(prompt_events)
    transposed_prompt_events = transpose_events(prompt_events, transpose_shift)
    score_tokens = events_to_score_tokens(transposed_prompt_events)
    prime = build_prime_tokens(score_tokens, resolved_seq_len, resolved_pad_idx)
    prime_tensor = torch.tensor(prime, dtype=torch.long, device=resolved_device)
    eos_token = resolve_eos_token(resolved_pad_idx)

    last_time_ms = 0.0
    if events:
        last_time_ms = max(float(e.get("time", 0.0)) for e in events) * 1000.0

    input_velocity = estimate_input_velocity(prompt_events)
    best_events: list[dict] = []
    best_tokens: list[int] = []
    best_score = -1e9

    candidate_count = max(1, min(6, int(num_candidates)))
    generation_started = time.perf_counter()
    candidate_timings_ms: list[float] = []
    for idx in range(candidate_count):
        if seed is not None:
            sample_seed = int(seed) + idx
            torch.manual_seed(sample_seed)
            if resolved_device == "cuda":
                torch.cuda.manual_seed_all(sample_seed)

        candidate_started = time.perf_counter()
        tokens = generate_tokens_sample(
            model,
            prime_tensor,
            generate_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token=eos_token,
        )
        if resolved_device == "cuda":
            torch.cuda.synchronize()
        candidate_timings_ms.append((time.perf_counter() - candidate_started) * 1000.0)
        new_tokens = tokens[len(prime) :]

        candidate_events = tokens_to_events(
            new_tokens,
            offset_ms=last_time_ms,
            velocity=input_velocity,
        )
        candidate_events = transpose_events(candidate_events, -transpose_shift)
        candidate_score = score_candidate_events(candidate_events, prompt_events)

        if candidate_score > best_score or idx == 0:
            best_score = candidate_score
            best_events = candidate_events
            best_tokens = new_tokens

    generation_total_ms = (time.perf_counter() - generation_started) * 1000.0
    avg_candidate_ms = (
        sum(candidate_timings_ms) / len(candidate_timings_ms)
        if candidate_timings_ms
        else 0.0
    )
    print(
        "Godzilla generation timing (model load excluded):",
        {
            "device": resolved_device,
            "generate_tokens": generate_tokens,
            "candidates": candidate_count,
            "candidate_ms": [round(ms, 2) for ms in candidate_timings_ms],
            "avg_candidate_ms": round(avg_candidate_ms, 2),
            "total_generation_ms": round(generation_total_ms, 2),
        },
    )

    return best_events, best_tokens


def generate_godzilla_continuation(
    events: list[dict],
    *,
    generate_tokens: int = 32,
    seed: int | None = None,
    temperature: float = 0.9,
    top_p: float = 0.95,
    num_candidates: int = 1,
    device: str = "auto",
    request: Any | None = None,
) -> tuple[list[dict], list[int]]:
    return generate_from_events(
        events,
        generate_tokens=generate_tokens,
        seed=seed,
        temperature=temperature,
        top_p=top_p,
        num_candidates=num_candidates,
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
        temperature: float = 0.9,
        top_p: float = 0.95,
        num_candidates: int = 1,
        request: Any | None = None,
        device: str = "auto",
    ) -> list[dict]:
        new_events, _ = generate_godzilla_continuation(
            events,
            generate_tokens=tokens,
            seed=seed,
            temperature=temperature,
            top_p=top_p,
            num_candidates=num_candidates,
            device=device,
            request=request,
        )
        return new_events


def get_model(model_id: str) -> MidiModel:
    if model_id == "godzilla":
        return GodzillaMidiModel()
    raise ValueError(f"Unknown MIDI model: {model_id}")
