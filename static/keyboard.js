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

// Computer keyboard mapping (fallback)
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
  'j': 71,  // B4
  'k': 72,  // C5
  'o': 73,  // C#5
  'l': 74,  // D5
  'p': 75,  // D#5
  ';': 76   // E5
};

// Keyboard shortcuts displayed on keys (fallback)
const keyShortcuts = {
  60: 'A', 61: 'W', 62: 'S', 63: 'E', 64: 'D', 65: 'F',
  66: 'T', 67: 'G', 68: 'Y', 69: 'H', 70: 'U', 71: 'J',
  72: 'K', 73: 'O', 74: 'L', 75: 'P', 76: ';'
};

// =============================================================================
// DOM ELEMENTS
// =============================================================================

const keyboardEl = document.getElementById('keyboard');
const statusEl = document.getElementById('status');
const recordBtn = document.getElementById('recordBtn');
const stopBtn = document.getElementById('stopBtn');
const playbackBtn = document.getElementById('playbackBtn');
const saveBtn = document.getElementById('saveBtn');
const panicBtn = document.getElementById('panicBtn');
const keyboardToggle = document.getElementById('keyboardToggle');
const instrumentSelect = document.getElementById('instrumentSelect');
const engineSelect = document.getElementById('engineSelect');
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
let selectedEngine = 'parrot'; // Default engine
let serverConfig = null; // Will hold instruments and keyboard config from server

// =============================================================================
// INSTRUMENT FACTORY
// =============================================================================

function buildInstruments(instrumentConfigs) {
  /**
   * Build Tone.js synth instances from config
   * instrumentConfigs: Object from server with instrument definitions
   */
  const instruments = {};
  
  for (const [key, config] of Object.entries(instrumentConfigs)) {
    const baseOptions = {
      maxPolyphony: 24,
      oscillator: config.oscillator ? { type: config.oscillator } : undefined,
      envelope: config.envelope,
    };
    
    // Remove undefined keys
    Object.keys(baseOptions).forEach(k => baseOptions[k] === undefined && delete baseOptions[k]);
    
    if (config.type === 'FMSynth') {
      baseOptions.harmonicity = config.harmonicity;
      baseOptions.modulationIndex = config.modulationIndex;
      instruments[key] = () => new Tone.PolySynth(Tone.FMSynth, baseOptions).toDestination();
    } else {
      instruments[key] = () => new Tone.PolySynth(Tone.Synth, baseOptions).toDestination();
    }
  }
  
  return instruments;
}

let instruments = {}; // Will be populated after config is fetched

function populateEngineSelect(engines) {
  if (!engineSelect || !Array.isArray(engines)) return;

  engineSelect.innerHTML = '';
  engines.forEach(engine => {
    const option = document.createElement('option');
    option.value = engine.id;
    option.textContent = engine.name || engine.id;
    engineSelect.appendChild(option);
  });

  if (engines.length > 0) {
    selectedEngine = engines[0].id;
    engineSelect.value = selectedEngine;
  }
}

// =============================================================================
// INITIALIZATION FROM SERVER CONFIG
// =============================================================================

async function initializeFromConfig() {
  /**
   * Fetch configuration from Python server and initialize UI
   */
  try {
    const response = await fetch('/gradio_api/api/config');
    if (!response.ok) throw new Error(`Config fetch failed: ${response.status}`);
    
    serverConfig = await response.json();
    
    // Build instruments from config
    instruments = buildInstruments(serverConfig.instruments);
    
    // Build keyboard shortcut maps from server config
    window.keyboardShortcutsFromServer = serverConfig.keyboard_shortcuts;
    window.keyMapFromServer = {};
    for (const [midiStr, key] of Object.entries(serverConfig.keyboard_shortcuts)) {
      window.keyMapFromServer[key.toLowerCase()] = parseInt(midiStr);
    }

    // Populate engine dropdown from server config
    populateEngineSelect(serverConfig.engines);
    
    // Render keyboard after config is loaded
    buildKeyboard();
    
  } catch (error) {
    console.error('Failed to load configuration:', error);
    // Fallback: Use hardcoded values for development/debugging
    console.warn('Using fallback hardcoded configuration');
    instruments = buildInstruments({
      'synth': {name: 'Synth', type: 'Synth', oscillator: 'sine', envelope: {attack: 0.005, decay: 0.1, sustain: 0.3, release: 0.2}},
      'piano': {name: 'Piano', type: 'Synth', oscillator: 'triangle', envelope: {attack: 0.001, decay: 0.2, sustain: 0.1, release: 0.3}},
      'organ': {name: 'Organ', type: 'Synth', oscillator: 'sine4', envelope: {attack: 0.001, decay: 0, sustain: 1, release: 0.1}},
      'bass': {name: 'Bass', type: 'Synth', oscillator: 'sawtooth', envelope: {attack: 0.01, decay: 0.1, sustain: 0.4, release: 0.3}},
      'pluck': {name: 'Pluck', type: 'Synth', oscillator: 'triangle', envelope: {attack: 0.001, decay: 0.3, sustain: 0, release: 0.3}},
      'fm': {name: 'FM', type: 'FMSynth', harmonicity: 3, modulationIndex: 10, envelope: {attack: 0.01, decay: 0.2, sustain: 0.2, release: 0.2}}
    });
    window.keyboardShortcutsFromServer = keyShortcuts; // Use hardcoded as fallback
    window.keyMapFromServer = keyMap; // Use hardcoded as fallback
    populateEngineSelect([
      { id: 'parrot', name: 'Parrot' },
      { id: 'reverse_parrot', name: 'Reverse Parrot' },
      { id: 'godzilla_continue', name: 'Godzilla' }
    ]);
    buildKeyboard();
  }
}

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
  // Clear any existing keys
  keyboardEl.innerHTML = '';
  
  for (let octave = 0; octave < numOctaves; octave++) {
    for (let i = 0; i < keys.length; i++) {
      const k = keys[i];
      const midiNote = baseMidi + (octave * 12) + k.offset;
      const octaveNum = 4 + octave;
      const keyEl = document.createElement('div');
      keyEl.className = 'key' + (k.black ? ' black' : '');
      keyEl.dataset.midi = midiNote;
      
      // Use server config shortcuts if available, otherwise fallback to hardcoded
      const shortcutsMap = window.keyboardShortcutsFromServer || keyShortcuts;
      const shortcut = shortcutsMap[midiNote] || '';
      const shortcutHtml = shortcut ? `<div class="shortcut-hint">${shortcut}</div>` : '';
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
  logToTerminal('╔═══════════════════════════════════════════════════════╗', 'timestamp');
  logToTerminal('║         🎹 MIDI MONITOR INITIALIZED 🎹              ║', 'timestamp');
  logToTerminal('╚═══════════════════════════════════════════════════════╝', 'timestamp');
  logToTerminal('Ready to capture MIDI events...', 'timestamp');
  logToTerminal('', '');
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
  playbackBtn.disabled = true;
  saveBtn.disabled = true;
  
  logToTerminal('', '');
  logToTerminal('▶▶▶ RECORDING STARTED ◀◀◀', 'timestamp');
  logToTerminal('', '');
}

function stopRecord() {
  recording = false;
  statusEl.textContent = `Recorded ${events.length} events`;
  recordBtn.disabled = false;
  stopBtn.disabled = true;
  saveBtn.disabled = events.length === 0;
  playbackBtn.disabled = events.length === 0;
  
  logToTerminal('', '');
  logToTerminal(`■■■ RECORDING STOPPED (${events.length} events captured) ■■■`, 'timestamp');
  logToTerminal('', '');
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
    const event = {
      type: 'note_on',
      note: midiNote,
      velocity: Math.max(1, velocity | 0),
      time: nowSec() - startTime,
      channel: 0
    };
    events.push(event);
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
    const event = {
      type: 'note_off',
      note: midiNote,
      velocity: 0,
      time: nowSec() - startTime,
      channel: 0
    };
    events.push(event);
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
  const keyMap = window.keyMapFromServer || keyMap; // Use server config if available, fallback to hardcoded
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
  const keyMap = window.keyMapFromServer || keyMap; // Use server config if available, fallback to hardcoded
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
    
    k.addEventListener('pointerdown', async (ev) => {
      ev.preventDefault();
      k.setPointerCapture(ev.pointerId);
      if (!pressed) {
        pressed = true;
        k.style.filter = 'brightness(0.85)';
        
        // Ensure Tone.js audio context is started
        await Tone.start();
        
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
  if (keyboardToggle.checked) {
    // Show keyboard shortcuts
    keyboardEl.classList.add('shortcuts-visible');
  } else {
    // Hide keyboard shortcuts
    keyboardEl.classList.remove('shortcuts-visible');
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
  logToTerminal('╔═══════════════════════════════════════════════════════╗', 'timestamp');
  logToTerminal('║         🎹 MIDI MONITOR INITIALIZED 🎹              ║', 'timestamp');
  logToTerminal('╚═══════════════════════════════════════════════════════╝', 'timestamp');
  logToTerminal('Ready to capture MIDI events...', 'timestamp');
  logToTerminal('', '');
});

recordBtn.addEventListener('click', async () => {
  await Tone.start();
  beginRecord();
});

stopBtn.addEventListener('click', () => stopRecord());

engineSelect.addEventListener('change', (e) => {
  selectedEngine = e.target.value;
  logToTerminal(`Engine switched to: ${selectedEngine}`, 'timestamp');
});

playbackBtn.addEventListener('click', async () => {
  if (events.length === 0) return alert('No recording to play back');
  
  // Ensure all notes are off before starting playback
  if (synth) {
    synth.releaseAll();
  }
  keyboardEl.querySelectorAll('.key').forEach(k => {
    k.style.filter = '';
  });
  
  statusEl.textContent = 'Playing back...';
  playbackBtn.disabled = true;
  recordBtn.disabled = true;
  
  logToTerminal('', '');
  logToTerminal('♫♫♫ PLAYBACK STARTED ♫♫♫', 'timestamp');
  logToTerminal('', '');
  
  try {
    // Process events through the selected engine
    let processedEvents = events;
    const selectedEngine = engineSelect.value;
    
    if (selectedEngine && selectedEngine !== 'parrot') {
      // Step 1: Start the engine processing call
      const startResp = await fetch('/gradio_api/call/process_engine', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          data: [{
            engine_id: selectedEngine,
            events: events
          }]
        })
      });
      
      if (!startResp.ok) {
        console.error('Engine API start failed:', startResp.status);
      } else {
        const startJson = await startResp.json();

        
        // Step 2: Poll for the result
        if (startJson && startJson.event_id) {
          const resultResp = await fetch(`/gradio_api/call/process_engine/${startJson.event_id}`);
          if (resultResp.ok) {
            const resultText = await resultResp.text();
            const dataLine = resultText.split('\n').find(line => line.startsWith('data:'));
            if (dataLine) {
              const payloadList = JSON.parse(dataLine.replace('data:', '').trim());
              const result = Array.isArray(payloadList) ? payloadList[0] : null;
              
              if (result && result.events) {
                processedEvents = result.events;
              }
            }
          }
        }
      }
    }
    
    // Play back the recorded events
    statusEl.textContent = 'Playing back...';
    let eventIndex = 0;
    
    const playEvent = () => {
      if (eventIndex >= processedEvents.length) {
        // Playback complete - ensure all notes are off
        if (synth) {
          synth.releaseAll();
        }
        
        // Clear all key highlights
        keyboardEl.querySelectorAll('.key').forEach(k => {
          k.style.filter = '';
        });
        
        statusEl.textContent = 'Playback complete';
        playbackBtn.disabled = false;
        recordBtn.disabled = false;
        
        logToTerminal('', '');
        logToTerminal('♫♫♫ PLAYBACK FINISHED ♫♫♫', 'timestamp');
        logToTerminal('', '');
        return;
      }
      
      const event = processedEvents[eventIndex];
      const nextTime = eventIndex + 1 < processedEvents.length
        ? processedEvents[eventIndex + 1].time
        : event.time;
      
      if (event.type === 'note_on') {
        const freq = Tone.Frequency(event.note, "midi").toFrequency();
        synth.triggerAttack(freq, undefined, event.velocity / 127);
        
        const noteName = midiToNoteName(event.note);
        logToTerminal(
          `[${event.time.toFixed(3)}s] ► ${noteName} (${event.note})`,
          'note-on'
        );
        
        // Highlight the key being played
        const keyEl = getKeyElement(event.note);
        if (keyEl) keyEl.style.filter = 'brightness(0.7)';
      } else if (event.type === 'note_off') {
        const freq = Tone.Frequency(event.note, "midi").toFrequency();
        synth.triggerRelease(freq);
        
        const noteName = midiToNoteName(event.note);
        logToTerminal(
          `[${event.time.toFixed(3)}s] ◄ ${noteName}`,
          'note-off'
        );
        
        // Remove key highlight
        const keyEl = getKeyElement(event.note);
        if (keyEl) keyEl.style.filter = '';
      }
      
      eventIndex++;
      const deltaTime = Math.max(0, nextTime - event.time);
      setTimeout(playEvent, deltaTime * 1000);
    };
    
    playEvent();
  } catch (err) {
    console.error('Playback error:', err);
    statusEl.textContent = 'Playback error: ' + err.message;
    playbackBtn.disabled = false;
    recordBtn.disabled = false;
    
    // Ensure all notes are off on error
    if (synth) {
      synth.releaseAll();
    }
    keyboardEl.querySelectorAll('.key').forEach(k => {
      k.style.filter = '';
    });
  }
});

saveBtn.addEventListener('click', () => saveMIDI());

panicBtn.addEventListener('click', () => {
  // Stop all notes immediately
  if (synth) {
    synth.releaseAll();
  }
  
  // Clear all pressed keys
  pressedKeys.clear();
  
  // Reset all visual key highlights
  keyboardEl.querySelectorAll('.key').forEach(k => {
    k.style.filter = '';
  });
  
  logToTerminal('🚨 PANIC - All notes stopped', 'timestamp');
});

// =============================================================================
// =============================================================================
// INITIALIZATION
// =============================================================================

async function init() {
  // First, load configuration from server
  await initializeFromConfig();
  
  // Then load default instrument (synth)
  loadInstrument('synth');
  
  // Setup keyboard event listeners and UI
  attachPointerEvents();
  initTerminal();
  
  // Set initial button states
  recordBtn.disabled = false;
  stopBtn.disabled = true;
  saveBtn.disabled = true;
  playbackBtn.disabled = true;
}

// Start the application when DOM is ready
document.addEventListener('DOMContentLoaded', init);
