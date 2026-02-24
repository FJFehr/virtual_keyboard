#!/usr/bin/env python3
"""
CPU vs GPU generation benchmark for the Godzilla MIDI model.
Sweeps all combinations of input length x generation length.

Usage:
    uv run python benchmark.py
    uv run python benchmark.py --runs 5 --candidates 1 --cpu-only
"""

import argparse
import datetime
import io
import math
import sys
import time
import torch
from midi_model import generate_godzilla_continuation

# Short input: 8 notes, 0.5s apart (~4 seconds, ~24 prompt tokens)
SHORT_EVENTS = [
    {
        "type": "note",
        "note": 60 + (i % 12),
        "velocity": 80,
        "time": i * 0.5,
        "channel": 0,
    }
    for i in range(8)
]

# Long input: 90 notes, 0.2s apart (~18 seconds — fills the prompt window)
LONG_EVENTS = [
    {
        "type": "note",
        "note": 60 + (i % 12),
        "velocity": 80,
        "time": i * 0.2,
        "channel": 0,
    }
    for i in range(90)
]

INPUT_FIXTURES = {
    "short (8 notes, ~4s)": SHORT_EVENTS,
    "long  (90 notes, ~18s)": LONG_EVENTS,
}

# Matches the four UI presets in keyboard.js
GENERATION_LENGTHS = [32, 64, 96, 128]


def gpu_name() -> str:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return "N/A"


def stddev(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    mean = sum(values) / n
    return math.sqrt(sum((x - mean) ** 2 for x in values) / (n - 1))


def run_generation(
    events: list[dict], device: str, tokens: int, candidates: int
) -> float:
    """Run one generation call, return wall-clock time in ms."""
    t0 = time.perf_counter()
    generate_godzilla_continuation(
        events,
        generate_tokens=tokens,
        device=device,
        num_candidates=candidates,
        seed=42,
    )
    return (time.perf_counter() - t0) * 1000.0


def benchmark_device(
    device: str, runs: int, candidates: int
) -> dict[tuple[str, int], list[float]]:
    """Run all input x generation-length combinations for one device."""
    print(f"\n{'=' * 72}")
    print(f"  Device: {device.upper()}  |  candidates={candidates}")
    print(f"{'=' * 72}")

    # Single warm-up to load the model (use smallest combo)
    print("  [warm-up] loading model + first inference...")
    run_generation(SHORT_EVENTS, device, GENERATION_LENGTHS[0], candidates)

    results: dict[tuple[str, int], list[float]] = {}
    for input_label, events in INPUT_FIXTURES.items():
        for gen_tokens in GENERATION_LENGTHS:
            key = (input_label, gen_tokens)
            timings = []
            print(
                f"  input={input_label}  gen={gen_tokens:>3} tokens",
                end="  ",
                flush=True,
            )
            for i in range(runs):
                ms = run_generation(events, device, gen_tokens, candidates)
                timings.append(ms)
                print(f"[{i + 1}:{ms:.0f}ms]", end=" ", flush=True)
            print()
            results[key] = timings

    return results


def print_summary(
    device: str, results: dict[tuple[str, int], list[float]], candidates: int
) -> None:
    print(f"\n{'=' * 80}")
    print(f"  SUMMARY — {device.upper()}  |  candidates={candidates}")
    print(f"{'=' * 80}")
    header = f"  {'Input':<24}  {'Gen tok':>7}  {'Mean ms':>8}  {'Mean s':>7}  {'Std ms':>7}  {'Min ms':>7}  {'Max ms':>7}  {'tok/s':>7}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for (input_label, gen_tokens), timings in results.items():
        mean = sum(timings) / len(timings)
        std = stddev(timings)
        tok_per_s = gen_tokens / (mean / 1000.0)
        print(
            f"  {input_label:<24}  {gen_tokens:>7}  {mean:>8.0f}  {mean / 1000:>7.2f}"
            f"  {std:>7.1f}  {min(timings):>7.0f}  {max(timings):>7.0f}  {tok_per_s:>7.1f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--candidates", type=int, default=1)
    parser.add_argument("--output", type=str, default="benchmark_results.txt")
    parser.add_argument("--cpu-only", action="store_true", help="Skip GPU benchmark")
    args = parser.parse_args()

    # Tee all output to stdout and a buffer for saving
    buffer = io.StringIO()

    class Tee:
        def write(self, msg):
            sys.__stdout__.write(msg)
            buffer.write(msg)

        def flush(self):
            sys.__stdout__.flush()

    sys.stdout = Tee()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Benchmark run: {timestamp}")
    print(f"GPU: {gpu_name()}")
    print(f"Runs per combination: {args.runs}  |  Candidates: {args.candidates}")
    print(
        f"Input sizes:      short={len(SHORT_EVENTS)} notes, long={len(LONG_EVENTS)} notes"
    )
    print(f"Generation sizes: {GENERATION_LENGTHS} tokens")

    all_results: dict[str, dict[tuple[str, int], list[float]]] = {}

    all_results["cpu"] = benchmark_device("cpu", args.runs, args.candidates)

    if args.cpu_only:
        print("\n[--cpu-only flag set — skipping GPU benchmark]")
    elif torch.cuda.is_available():
        all_results["cuda"] = benchmark_device("cuda", args.runs, args.candidates)
    else:
        print("\n[CUDA not available — skipping GPU benchmark]")

    for device, results in all_results.items():
        print_summary(device, results, args.candidates)

    # GPU speedup table (if both ran)
    if "cpu" in all_results and "cuda" in all_results:
        print(f"\n{'=' * 80}")
        print("  GPU SPEEDUP")
        print(f"{'=' * 80}")
        header = f"  {'Input':<24}  {'Gen tok':>7}  {'CPU ms':>8}  {'CPU s':>6}  {'GPU ms':>8}  {'GPU s':>6}  {'Speedup':>8}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for key in all_results["cpu"]:
            cpu_mean = sum(all_results["cpu"][key]) / len(all_results["cpu"][key])
            gpu_mean = sum(all_results["cuda"][key]) / len(all_results["cuda"][key])
            speedup = cpu_mean / gpu_mean
            input_label, gen_tokens = key
            print(
                f"  {input_label:<24}  {gen_tokens:>7}  {cpu_mean:>8.0f}  {cpu_mean / 1000:>6.2f}"
                f"  {gpu_mean:>8.0f}  {gpu_mean / 1000:>6.2f}  {speedup:>7.2f}x"
            )

    print()
    sys.stdout = sys.__stdout__

    with open(args.output, "w") as f:
        f.write(buffer.getvalue())
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
