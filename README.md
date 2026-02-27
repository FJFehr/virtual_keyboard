---
title: SYNTHIA
emoji: 🎹
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
short_description: Browser-based MIDI keyboard with recording and synthesis
---

# SYNTHIA

Play, record, and let AI continue your musical phrases in real-time. 🎹

## � Quick Start

```bash
# Install dependencies
uv sync

# Run the app
uv run python app.py
```

Open **http://127.0.0.1:7860**

---

## 🏗️ Architecture Overview

**SYNTHIA** is a browser-based MIDI keyboard with three main layers:

1. **Backend** (Python/Gradio): Configuration, MIDI engines, model loading
2. **Frontend** (JavaScript/Tone.js): Audio synthesis, keyboard rendering, event handling
3. **Communication**: Gradio bridge for sending recorded MIDI to backend for processing

### Data Flow

```
User plays keyboard
    ↓
JavaScript captures MIDI events → records to array
    ↓
User clicks "Play/Process"
    ↓
Backend engine processes recorded events
    ↓
Result returned as MIDI events
    ↓
JavaScript plays result through Tone.js synth
```

---

## 📂 File Responsibilities

### Backend Files

| File | Purpose |
|------|---------|
| **app.py** | Gradio app setup, UI layout, instrument definitions, API endpoints |
| **config.py** | Global settings (audio parameters, model paths, inference defaults) |
| **engines.py** | Three MIDI processing engines: `parrot` (repeat), `reverse_parrot` (reverse), `godzilla_continue` (AI generation) |
| **midi_model.py** | Godzilla model loading, tokenization, inference |
| **midi.py** | MIDI file utilities (encode/decode, cleanup, utilities) |

### Frontend Files

| File | Purpose |
|------|---------|
| **keyboard.html** | DOM structure (keyboard grid, controls, terminal) |
| **keyboard.js** | Main application logic: keyboard rendering, audio synthesis (Tone.js), recording, UI event binding, engine communication |
| **styles.css** | Styling and animations |

### Configuration & Dependencies

| File | Purpose |
|------|---------|
| **requirements.txt** | Python dependencies |
| **pyproject.toml** | Project metadata |

---

## 🎹 Core Functionality

### Keyboard Controls
- **Click keys** or **press computer keys** to play notes
- **Record button**: Capture MIDI events from keyboard
- **Play button**: Play back recorded events
- **Save button**: Download recording as .mid file
- **Game mode**: Beat-synced call-and-response loop with fixed 75 BPM metronome

### MIDI Engines
1. **Parrot**: Repeats your exact melody
2. **Reverse Parrot**: Plays melody backward
3. **Godzilla**: AI generates musical continuations using transformer model

### UI Features
- **Engine selector**: Choose processing method
- **Style selector**: AI style (melodic, energetic, ornate, etc.)
- **Response mode**: Control AI generation behavior
- **Runtime selector**: GPU (fast) vs CPU (reliable)
- **User Bars / AI Bars**: Independent 1- or 2-bar turn lengths in game mode
- **Instrument selector**: Change synth sound
- **AI voice selector**: Change AI synth sound
- **Terminal**: Real-time event logging

### Beat-Synced Game Mode
- **Fixed tempo**: 75 BPM, continuous 4/4 metronome while game mode is active
- **Count-in**: Animated `3, 2, 1, GO` before user and AI turns
- **Turn windows**: User and AI each use independently selected bar lengths (1 or 2)
- **Quantized AI playback**: AI response is quantized to a 16th-note grid and clamped to the selected AI bar window
- **Loop behavior**: Empty user turns automatically retry after a fresh count-in; loop continues until Stop Game

---

## 🔧 How to Add New Functionality

### Adding a New MIDI Engine

1. **In `engines.py`**, add a new function:
   ```python
   def my_new_engine(events, options):
       # Process MIDI events
       return processed_events
   ```

2. **In `app.py`**, register the engine in `process_events()`:
   ```python
   elif engine == 'my_engine':
       result_events = my_new_engine(events, options)
   ```

3. **In `app.py`**, add to engine dropdown:
   ```python
   with gr.Group(label="Engine"):
       engine = gr.Dropdown(
           choices=['parrot', 'reverse_parrot', 'godzilla_continue', 'my_engine'],
           # ...
       )
   ```

4. **In `keyboard.js`**, add tooltip (line ~215 in `populateEngineSelect()`):
   ```javascript
   const engineTooltips = {
       'my_engine': 'Description of what your engine does'
   };
   ```

### Adding a New Control Selector

1. **In `app.py`**, create the selector in the UI:
   ```python
   my_control = gr.Dropdown(
       choices=['option1', 'option2'],
       label="My Control",
       value='option1'
   )
   ```

2. **In `keyboard.js`** (line ~1510), add to `selectControls` array:
   ```javascript
   {
       element: myControlSelect,
       getter: () => ({ label: myControlSelect.value }),
       message: (result) => `Control switched to: ${result.label}`
   }
   ```

3. **In `keyboard.js`**, pass control to engine via `processEventsThroughEngine()`:
   ```javascript
   const engineOptions = {
       my_control: document.getElementById('myControl').value,
       // ... other options
   };
   ```

### Adding a New Response Mode

1. **In `keyboard.js`** (line ~175), add preset definition:
   ```javascript
   const RESPONSE_MODES = {
       'my_mode': {
           label: 'My Mode',
           processFunction: (events) => {
               // Processing logic
               return processedEvents;
           }
       }
   };
   ```

2. **In `app.py`**, add to response mode dropdown

3. **Use in engine logic** via `getSelectedResponseMode()`

---

## 🔄 Recent Refactoring (Feb 2026)

Code consolidation to improve maintainability:

- **Consolidated getter functions**: Single `getSelectedPreset()` replaces 3 similar functions
- **Unified event listeners**: Loop-based pattern for select controls (runtime, style, mode, length)
- **Extracted helper functions**: `resetAllNotesAndVisuals()` replaces 3 duplicated blocks
- **Result**: Reduced redundancy, easier to modify preset logic, consistent patterns

---

## ⚡ Benchmarking

`benchmark.py` measures Godzilla model generation speed across all combinations of input length and generation length, with CPU and GPU compared side by side.

### What it tests

| Axis | Values |
|------|--------|
| Input length | Short (8 notes, ~4 s) · Long (90 notes, ~18 s) |
| Generation length | 32 · 64 · 96 · 128 tokens (matches the four UI presets) |
| Devices | CPU always · CUDA if available |

Each combination runs a warm-up pass (model load, timing discarded) followed by `--runs` timed passes. The summary tables report mean, std, min, max in both ms and seconds, plus tokens/sec and GPU speedup.

### Usage

```bash
# Full sweep — CPU + GPU (if available), 5 runs per combination
uv run python benchmark.py

# CPU only (useful for verifying the script or on CPU-only machines)
uv run python benchmark.py --cpu-only

# Increase runs for tighter statistics
uv run python benchmark.py --runs 10

# Multi-candidate generation (higher quality, slower)
uv run python benchmark.py --candidates 3
```

Results are printed to stdout and saved to `benchmark_results.txt` (override with `--output`).

### Example output

```
============================================================
  Device: CUDA  |  candidates=1
============================================================
  [warm-up] loading model + first inference...
  input=short (8 notes, ~4s)   gen= 32 tokens  [1:85ms] [2:82ms] ...
  ...

================================================================================
  SUMMARY — CUDA  |  candidates=1
================================================================================
  Input                     Gen tok   Mean ms    Mean s   Std ms   Min ms   Max ms   tok/s
  -----------------------------------------------------------------------------------------
  short (8 notes, ~4s)           32        85      0.09      2.1       82       89   376.5
  short (8 notes, ~4s)          128       290      0.29      4.3      284      297   441.4
  long  (90 notes, ~18s)         32        91      0.09      1.8       88       94   351.6
  long  (90 notes, ~18s)        128       305      0.31      3.9      299      312   419.7
```

---

## 🛠️ Development Tips

### Debugging
- **Terminal in UI**: Shows all MIDI events and engine responses
- **Browser console**: `F12` for JavaScript errors
- **Python terminal**: Check server-side logs for model loading, inference errors

### Testing New Engines
1. Record a simple 3-5 note progression
2. Play back with different engines
3. Check terminal for processing details
4. Verify output notes are in valid range (0-127)

### Performance
- **Recording**: Event capture happens in JavaScript (fast, local)
- **Processing**: May take 2-5 seconds depending on engine and model
- **Playback**: Tone.js synthesis is real-time (instant)

---

## 🔧 Technology Stack

- **Frontend**: Tone.js v6+ (Web Audio API)
- **Backend**: Gradio 5.49.1 + Python 3.10+
- **MIDI**: mido library
- **Model**: Godzilla Piano Transformer (via Hugging Face)

---

## 📝 License

Open source - free to use and modify.
