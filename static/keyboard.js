/**
 * Virtual MIDI Keyboard - Main JavaScript
 * 
 * This file handles:
 * - Keyboard rendering and layout
 * - Audio synthesis (Tone.js)
 * - MIDI event recording
 * - Computer keyboard input
 * - MIDI monitor/terminal
 * - File export
 */

// =============================================================================
// CONFIGURATION
// =============================================================================

const baseMidi = 60; // C4
const numOctaves = 2;

// Keyboard layout with sharps flagged
const keys = [
  {name:'C',  offset:0,  black:false},
  {name:'C#', offset:1,  black:true},
  {name:'D',  offset:2,  black:false},
  {name:'D#', offset:3,  black:true},
  {name:'E',  offset:4,  black:false},
  {name:'F',  offset:5,  black:false},
  {name:'F#', offset:6,  black:true},
  {name:'G',  offset:7,  black:false},
  {name:'G#', offset:8,  black:true},
  {name:'A',  offset:9,  black:false},
  {name:'A#', offset:10, black:true},
  {name:'B',  offset:11, black:false}
];

// Computer keyboard mapping (C4 octave)
const keyMap = {
  'a': 60,  // C4
  'w': 61,  // C#4
  's': 62,  // D4
  'e': 63,  // D#4
  'd': 64,  // E4
  'f': 65,  // F4
  't': 66,  // F#4
  'g': 67,  // G4
  'y': 68,  // G#4
  'h': 69,  // A4
  'u': 70,  // A#4
  'j': 71   // B4
};

// Keyboard shortcuts displayed on keys
const keyShortcuts = {
  60: 'A', 61: 'W', 62: 'S', 63: 'E', 64: 'D', 65: 'F',
  66: 'T', 67: 'G', 68: 'Y', 69: 'H', 70: 'U', 71: 'J'
};

// =============================================================================
// DOM ELEMENTS
// =============================================================================

const keyboardEl = document.getElementById('keyboard');
const statusEl = document.getElementById('status');
const recordBtn = document.getElementById('recordBtn');
const stopBtn = document.getElementById('stopBtn');
const saveBtn = document.getElementById('saveBtn');
const keyboardToggle = document.getElementById('keyboardToggle');
const instrumentSelect = document.getElementById('instrumentSelect');
const terminal = document.getElementById('terminal');
const clearTerminal = document.getElementById('clearTerminal');

// =============================================================================
// STATE
// =============================================================================

let synth = null;
let recording = false;
let startTime = 0;
let events = [];
const pressedKeys = new Set();

// =============================================================================
// INSTRUMENT CONFIGURATIONS
// =============================================================================

const instruments = {
  synth: () => new Tone.PolySynth(Tone.Synth, {
    oscillator: { type: 'sine' },
    envelope: { attack: 0.005, decay: 0.1, sustain: 0.3, release: 1 }
  }).toDestination(),
  
  piano: () => new Tone.PolySynth(Tone.Synth, {
    oscillator: { type: 'triangle' },
    envelope: { attack: 0.001, decay: 0.2, sustain: 0.1, release: 2 }
  }).toDestination(),
  
  organ: () => new Tone.PolySynth(Tone.Synth, {
    oscillator: { type: 'sine4' },
    envelope: { attack: 0.001, decay: 0, sustain: 1, release: 0.1 }
  }).toDestination(),
  
  bass: () => new Tone.PolySynth(Tone.Synth, {
    oscillator: { type: 'sawtooth' },
    envelope: { attack: 0.01, decay: 0.1, sustain: 0.4, release: 1.5 },
    filter: { Q: 2, type: 'lowpass', rolloff: -12 }
  }).toDestination(),
  
  pluck: () => new Tone.PolySynth(Tone.Synth, {
    oscillator: { type: 'triangle' },
    envelope: { attack: 0.001, decay: 0.3, sustain: 0, release: 0.3 }
  }).toDestination(),
  
  fm: () => new Tone.PolySynth(Tone.FMSynth, {
    harmonicity: 3,
    modulationIndex: 10,
    envelope: { attack: 0.01, decay: 0.2, sustain: 0.2, release: 1 }
  }).toDestination()
};

function loadInstrument(type) {
  if (synth) {
    synth.releaseAll();
    synth.dispose();
  }
  synth = instruments[type]();
}

// =============================================================================
// KEYBOARD RENDERING
// =============================================================================

function buildKeyboard() {
  for (let octave = 0; octave < numOctaves; octave++) {
    for (let i = 0; i < keys.length; i++) {
      const k = keys[i];
      const midiNote = baseMidi + (octave * 12) + k.offset;
      const octaveNum = 4 + octave;
      const keyEl = document.createElement('div');
      keyEl.className = 'key' + (k.black ? ' black' : '');
      keyEl.dataset.midi = midiNote;
      
      const shortcut = keyShortcuts[midiNote] || '';
      const shortcutHtml = shortcut ? `<div style="font-size:10px;opacity:0.5;">${shortcut}</div>` : '';
      keyEl.innerHTML = `<div style="padding-bottom:6px;font-size:11px">${shortcutHtml}${k.name}${octaveNum}</div>`;
      
      keyboardEl.appendChild(keyEl);
    }
  }
}

// =============================================================================
// MIDI UTILITIES
// =============================================================================

const noteNames = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'];

function midiToNoteName(midi) {
  const octave = Math.floor(midi / 12) - 1;
  const noteName = noteNames[midi % 12];
  return `${noteName}${octave}`;
}

function nowSec() {
  return performance.now() / 1000;
}

// =============================================================================
// TERMINAL LOGGING
// =============================================================================

function logToTerminal(message, className = '') {
  const line = document.createElement('div');
  line.className = className;
  line.textContent = message;
  terminal.appendChild(line);
  terminal.scrollTop = terminal.scrollHeight;
  
  // Keep terminal from getting too long (max 500 lines)
  while (terminal.children.length > 500) {
    terminal.removeChild(terminal.firstChild);
  }
}

function initTerminal() {
  logToTerminal('=== MIDI Monitor Ready ===', 'timestamp');
  logToTerminal('Play notes to see MIDI events...', 'timestamp');
}

// =============================================================================
// RECORDING 
// =============================================================================

function beginRecord() {
  events = [];
  recording = true;
  startTime = nowSec();
  statusEl.textContent = 'Recording...';
  recordBtn.disabled = true;
  stopBtn.disabled = false;
  saveBtn.disabled = true;
  logToTerminal('\n=== RECORDING STARTED ===', 'timestamp');
}

function stopRecord() {
  recording = false;
  statusEl.textContent = `Recorded ${events.length} events`;
  recordBtn.disabled = false;
  stopBtn.disabled = true;
  saveBtn.disabled = events.length === 0;
  logToTerminal(`=== RECORDING STOPPED (${events.length} events) ===\n`, 'timestamp');
}

// =============================================================================
// MIDI NOTE HANDLING
// =============================================================================

function noteOn(midiNote, velocity = 100) {
  const freq = Tone.Frequency(midiNote, "midi").toFrequency();
  synth.triggerAttack(freq, undefined, velocity / 127);
  
  const noteName = midiToNoteName(midiNote);
  const timestamp = recording ? (nowSec() - startTime).toFixed(3) : '--';
  logToTerminal(
    `[${timestamp}s] NOTE_ON  ${noteName} (${midiNote}) vel=${velocity}`, 
    'note-on'
  );
  
  if (recording) {
    events.push({
      type: 'note_on',
      note: midiNote,
      velocity: Math.max(1, velocity | 0),
      time: nowSec() - startTime,
      channel: 0
    });
  }
}

function noteOff(midiNote) {
  const freq = Tone.Frequency(midiNote, "midi").toFrequency();
  synth.triggerRelease(freq);
  
  const noteName = midiToNoteName(midiNote);
  const timestamp = recording ? (nowSec() - startTime).toFixed(3) : '--';
  logToTerminal(
    `[${timestamp}s] NOTE_OFF ${noteName} (${midiNote})`, 
    'note-off'
  );
  
  if (recording) {
    events.push({
      type: 'note_off',
      note: midiNote,
      velocity: 0,
      time: nowSec() - startTime,
      channel: 0
    });
  }
}

// =============================================================================
// COMPUTER KEYBOARD INPUT
// =============================================================================

function getKeyElement(midiNote) {
  return keyboardEl.querySelector(`.key[data-midi="${midiNote}"]`);
}

document.addEventListener('keydown', async (ev) => {
  if (!keyboardToggle.checked) return;
  const key = ev.key.toLowerCase();
  if (!keyMap[key] || pressedKeys.has(key)) return;
  
  ev.preventDefault();
  pressedKeys.add(key);
  
  await Tone.start();
  
  const midiNote = keyMap[key];
  const keyEl = getKeyElement(midiNote);
  if (keyEl) keyEl.style.filter = 'brightness(0.85)';
  noteOn(midiNote, 100);
});

document.addEventListener('keyup', (ev) => {
  if (!keyboardToggle.checked) return;
  const key = ev.key.toLowerCase();
  if (!keyMap[key] || !pressedKeys.has(key)) return;
  
  ev.preventDefault();
  pressedKeys.delete(key);
  
  const midiNote = keyMap[key];
  const keyEl = getKeyElement(midiNote);
  if (keyEl) keyEl.style.filter = '';
  noteOff(midiNote);
});

// =============================================================================
// MOUSE/TOUCH INPUT
// =============================================================================

function attachPointerEvents() {
  keyboardEl.querySelectorAll('.key').forEach(k => {
    let pressed = false;
    
    k.addEventListener('pointerdown', (ev) => {
      ev.preventDefault();
      k.setPointerCapture(ev.pointerId);
      if (!pressed) {
        pressed = true;
        k.style.filter = 'brightness(0.85)';
        const midi = parseInt(k.dataset.midi);
        const vel = ev.pressure ? Math.round(ev.pressure * 127) : 100;
        noteOn(midi, vel);
      }
    });
    
    k.addEventListener('pointerup', (ev) => {
      ev.preventDefault();
      if (pressed) {
        pressed = false;
        k.style.filter = '';
        const midi = parseInt(k.dataset.midi);
        noteOff(midi);
      }
    });
    
    k.addEventListener('pointerleave', (ev) => {
      if (pressed) {
        pressed = false;
        k.style.filter = '';
        const midi = parseInt(k.dataset.midi);
        noteOff(midi);
      }
    });
  });
}

// =============================================================================
// MIDI FILE EXPORT
// =============================================================================

async function saveMIDI() {
  if (recording) stopRecord();
  if (events.length === 0) return alert('No events recorded.');

  statusEl.textContent = 'Uploading…';
  saveBtn.disabled = true;
  
  try {
    const startResp = await fetch('/gradio_api/call/save_midi', {
      method: 'POST',
      headers: {'Content-Type':'application/json'},
      body: JSON.stringify({data: [events]})
    });
    
    if (!startResp.ok) {
      const txt = await startResp.text();
      throw new Error('Server error: ' + txt);
    }
    
    const startJson = await startResp.json();
    if (!startJson || !startJson.event_id) {
      throw new Error('Invalid API response');
    }
    
    const resultResp = await fetch(`/gradio_api/call/save_midi/${startJson.event_id}`);
    if (!resultResp.ok) {
      const txt = await resultResp.text();
      throw new Error('Server error: ' + txt);
    }
    
    const resultText = await resultResp.text();
    const dataLine = resultText.split('\n').find(line => line.startsWith('data:'));
    if (!dataLine) {
      throw new Error('Invalid API response');
    }
    
    const payloadList = JSON.parse(dataLine.replace('data:', '').trim());
    const payload = Array.isArray(payloadList) ? payloadList[0] : null;
    if (!payload || payload.error || !payload.midi_base64) {
      throw new Error(payload && payload.error ? payload.error : 'Invalid API response');
    }
    
    const binStr = atob(payload.midi_base64);
    const bytes = new Uint8Array(binStr.length);
    for (let i = 0; i < binStr.length; i++) {
      bytes[i] = binStr.charCodeAt(i);
    }
    
    const blob = new Blob([bytes], {type: 'audio/midi'});
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'recording.mid';
    a.click();
    statusEl.textContent = 'Downloaded .mid';
  } catch (err) {
    console.error(err);
    statusEl.textContent = 'Error saving MIDI';
    alert('Error: ' + err.message);
  } finally {
    saveBtn.disabled = false;
  }
}

// =============================================================================
// EVENT LISTENERS
// =============================================================================

instrumentSelect.addEventListener('change', () => {
  loadInstrument(instrumentSelect.value);
});

keyboardToggle.addEventListener('change', () => {
  if (!keyboardToggle.checked) {
    // Release all currently pressed keyboard keys
    pressedKeys.forEach(key => {
      const midiNote = keyMap[key];
      const keyEl = getKeyElement(midiNote);
      if (keyEl) keyEl.style.filter = '';
      noteOff(midiNote);
    });
    pressedKeys.clear();
  }
});

clearTerminal.addEventListener('click', () => {
  terminal.innerHTML = '';
  logToTerminal('=== MIDI Monitor Ready ===', 'timestamp');
});

recordBtn.addEventListener('click', async () => {
  await Tone.start();
  beginRecord();
});

stopBtn.addEventListener('click', () => stopRecord());

saveBtn.addEventListener('click', () => saveMIDI());

// =============================================================================
// INITIALIZATION
// =============================================================================

function init() {
  loadInstrument('synth');
  buildKeyboard();
  attachPointerEvents();
  initTerminal();
  
  // Set initial button states
  recordBtn.disabled = false;
  stopBtn.disabled = true;
  saveBtn.disabled = true;
}

// Start the application
init();
