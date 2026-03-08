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

let keyboardEl = null;
let statusEl = null;
let recordBtn = null;
let stopBtn = null;
let playbackBtn = null;
let gameStartBtn = null;
let gameStopBtn = null;
let saveBtn = null;
let panicBtn = null;
let keyboardToggle = null;
let instrumentSelect = null;
let aiInstrumentSelect = null;
let engineSelect = null;
let runtimeSelect = null;
let responseStyleSelect = null;
let responseModeSelect = null;
let responseLengthSelect = null;
let quantizationSelect = null;
let userBarsSelect = null;
let aiBarsSelect = null;
let terminal = null;
let clearTerminal = null;
let countdownOverlay = null;
let countdownText = null;
let userGridCanvas = null;
let aiGridCanvas = null;
let userGridMeta = null;
let aiGridMeta = null;
let gridPhaseBadge = null;

// =============================================================================
// STATE
// =============================================================================

let synth = null;
let aiSynth = null;
let recording = false;
let startTime = 0;
let events = [];
const pressedKeys = new Set();
let selectedEngine = 'parrot'; // Default engine
let serverConfig = null; // Will hold instruments and keyboard config from server
let gameActive = false;
let gameTurn = 0;

const GAME_BPM = 75;
const GAME_BEATS_PER_BAR = 4;
const GAME_COUNTIN_BEATS = 3;
const GAME_RETRY_DELAY_MS = 500;

let gameSessionId = 0;
let gameClockOriginSec = 0;
let gamePhase = 'idle';
let gameCaptureActive = false;
let gameCaptureStartWallSec = 0;
let gameCapturedEvents = [];
const gameCaptureActiveNotes = new Set();
const gameTimeoutIds = new Set();
let metronomeBeatIndex = 0;
let metronomeKick = null;
let metronomeSnare = null;
let metronomeHat = null;
let gameGridUserEvents = [];
let gameGridAIEvents = [];
let resolvedAutoRuntimeMode = null;
let autoRuntimeProbeInFlight = false;
let gridAnimationFrameId = null;
const gridPlayheads = {
  user: { active: false, startWallSec: 0, durationSec: 1 },
  ai: { active: false, startWallSec: 0, durationSec: 1 }
};

const RESPONSE_MODES = {
  raw_godzilla: { label: 'Raw Godzilla' },
  current_pipeline: { label: 'Current Pipeline' },
  musical_polish: { label: 'Musical Polish' }
};

const RESPONSE_LENGTH_PRESETS = {
  short: {
    label: 'Short',
    generateTokens: 32,
    maxNotes: 8,
    maxDurationSec: 4.0
  },
  medium: {
    label: 'Medium',
    generateTokens: 64,
    maxNotes: 14,
    maxDurationSec: 6.0
  },
  long: {
    label: 'Long',
    generateTokens: 96,
    maxNotes: 20,
    maxDurationSec: 8.0
  },
  extended: {
    label: 'Extended',
    generateTokens: 128,
    maxNotes: 28,
    maxDurationSec: 11.0
  }
};

const RESPONSE_STYLE_PRESETS = {
  melodic: {
    label: 'Melodic',
    maxNotes: 8,
    maxDurationSec: 4.0,
    smoothLeaps: true,
    addMotifEcho: false,
    playfulShift: false
  },
  motif_echo: {
    label: 'Motif Echo',
    maxNotes: 10,
    maxDurationSec: 4.3,
    smoothLeaps: true,
    addMotifEcho: true,
    playfulShift: false
  },
  playful: {
    label: 'Playful',
    maxNotes: 9,
    maxDurationSec: 3.8,
    smoothLeaps: true,
    addMotifEcho: false,
    playfulShift: true
  }
};

const GAME_QUANTIZATION_PRESETS = {
  sixteenth: {
    label: '16th Notes',
    stepBeats: 0.25
  },
  eighth: {
    label: '8th Notes',
    stepBeats: 0.5
  },
  none: {
    label: 'No Quantization',
    stepBeats: null
  }
};

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

  // Tooltip map for engine options
  const engineTooltips = {
    'parrot': 'Repeats your exact melody',
    'reverse_parrot': 'Plays your melody backward',
    'godzilla_continue': 'MIDI transformer'
  };

  engineSelect.innerHTML = '';
  engines.forEach(engine => {
    const option = document.createElement('option');
    option.value = engine.id;
    option.textContent = engine.name || engine.id;
    // Add tooltip attribute for hover display
    if (engineTooltips[engine.id]) {
      option.setAttribute('data-tooltip', engineTooltips[engine.id]);
    }
    engineSelect.appendChild(option);
  });

  if (engines.length > 0) {
    const hasGodzilla = engines.some(engine => engine.id === 'godzilla_continue');
    selectedEngine = hasGodzilla ? 'godzilla_continue' : engines[0].id;
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
    serverConfig = await callGradioBridge('config', {});
    if (!serverConfig || typeof serverConfig !== 'object') {
      throw new Error('Invalid config payload');
    }
    
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
    resetAutoRuntimeResolution();
    
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
    resetAutoRuntimeResolution();
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

function loadAIInstrument(type) {
  if (aiSynth) {
    aiSynth.releaseAll();
    aiSynth.dispose();
  }
  aiSynth = instruments[type]();
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

function getBridgeButton(buttonId) {
  return document.getElementById(buttonId) || document.querySelector(`#${buttonId} button`);
}

function getBridgeField(fieldId) {
  const root = document.getElementById(fieldId);
  if (!root) return null;
  if (root instanceof HTMLTextAreaElement || root instanceof HTMLInputElement) {
    return root;
  }
  return root.querySelector('textarea, input');
}

function setFieldValue(field, value) {
  const setter = field instanceof HTMLTextAreaElement
    ? Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value')?.set
    : Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')?.set;
  if (setter) {
    setter.call(field, value);
  } else {
    field.value = value;
  }
  field.dispatchEvent(new Event('input', { bubbles: true }));
  field.dispatchEvent(new Event('change', { bubbles: true }));
}

function waitForFieldUpdate(field, previousValue, timeoutMs = 120000) {
  return new Promise((resolve, reject) => {
    const deadline = Date.now() + timeoutMs;

    const check = () => {
      const nextValue = field.value || '';
      if (nextValue !== previousValue && nextValue !== '') {
        resolve(nextValue);
        return;
      }
      if (Date.now() > deadline) {
        reject(new Error('Timed out waiting for Gradio response'));
        return;
      }
      setTimeout(check, 80);
    };

    check();
  });
}

async function waitForBridgeElements(timeoutMs = 20000) {
  const required = [
    { kind: 'field', id: 'vk_config_input' },
    { kind: 'field', id: 'vk_config_output' },
    { kind: 'button', id: 'vk_config_btn' },
    { kind: 'field', id: 'vk_save_input' },
    { kind: 'field', id: 'vk_save_output' },
    { kind: 'button', id: 'vk_save_btn' },
    { kind: 'field', id: 'vk_engine_input' },
    { kind: 'field', id: 'vk_engine_cpu_output' },
    { kind: 'button', id: 'vk_engine_cpu_btn' },
    { kind: 'field', id: 'vk_engine_gpu_output' },
    { kind: 'button', id: 'vk_engine_gpu_btn' }
  ];

  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    const allReady = required.every(item => (
      item.kind === 'button'
        ? Boolean(getBridgeButton(item.id))
        : Boolean(getBridgeField(item.id))
    ));
    if (allReady) return;
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error('Gradio bridge elements were not ready in time');
}

function cacheUIElements() {
  keyboardEl = document.getElementById('keyboard');
  statusEl = document.getElementById('status');
  recordBtn = document.getElementById('recordBtn');
  stopBtn = document.getElementById('stopBtn');
  playbackBtn = document.getElementById('playbackBtn');
  gameStartBtn = document.getElementById('gameStartBtn');
  gameStopBtn = document.getElementById('gameStopBtn');
  saveBtn = document.getElementById('saveBtn');
  panicBtn = document.getElementById('panicBtn');
  keyboardToggle = document.getElementById('keyboardToggle');
  instrumentSelect = document.getElementById('instrumentSelect');
  aiInstrumentSelect = document.getElementById('aiInstrumentSelect');
  engineSelect = document.getElementById('engineSelect');
  runtimeSelect = document.getElementById('runtimeSelect');
  responseStyleSelect = document.getElementById('responseStyleSelect');
  responseModeSelect = document.getElementById('responseModeSelect');
  responseLengthSelect = document.getElementById('responseLengthSelect');
  quantizationSelect = document.getElementById('quantizationSelect');
  userBarsSelect = document.getElementById('userBarsSelect');
  aiBarsSelect = document.getElementById('aiBarsSelect');
  terminal = document.getElementById('terminal');
  clearTerminal = document.getElementById('clearTerminal');
  countdownOverlay = document.getElementById('countdownOverlay');
  countdownText = document.getElementById('countdownText');
  userGridCanvas = document.getElementById('userGridCanvas');
  aiGridCanvas = document.getElementById('aiGridCanvas');
  userGridMeta = document.getElementById('userGridMeta');
  aiGridMeta = document.getElementById('aiGridMeta');
  gridPhaseBadge = document.getElementById('gridPhaseBadge');
}

async function waitForKeyboardUIElements(timeoutMs = 20000) {
  const requiredIds = [
    'keyboard',
    'status',
    'recordBtn',
    'stopBtn',
    'playbackBtn',
    'gameStartBtn',
    'gameStopBtn',
    'saveBtn',
    'panicBtn',
    'keyboardToggle',
    'instrumentSelect',
    'engineSelect',
    'runtimeSelect',
    'quantizationSelect',
    'userBarsSelect',
    'aiBarsSelect',
    'terminal',
    'clearTerminal',
    'countdownOverlay',
    'countdownText',
    'userGridCanvas',
    'aiGridCanvas',
    'userGridMeta',
    'aiGridMeta',
    'gridPhaseBadge'
  ];

  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    const allReady = requiredIds.every(id => Boolean(document.getElementById(id)));
    if (allReady) return;
    await new Promise(resolve => setTimeout(resolve, 100));
  }
  throw new Error('Keyboard UI elements were not ready in time');
}

const BRIDGE_ACTIONS = {
  config: {
    inputId: 'vk_config_input',
    outputId: 'vk_config_output',
    buttonId: 'vk_config_btn'
  },
  save_midi: {
    inputId: 'vk_save_input',
    outputId: 'vk_save_output',
    buttonId: 'vk_save_btn'
  },
  process_engine_cpu: {
    inputId: 'vk_engine_input',
    outputId: 'vk_engine_cpu_output',
    buttonId: 'vk_engine_cpu_btn'
  },
  process_engine_gpu: {
    inputId: 'vk_engine_input',
    outputId: 'vk_engine_gpu_output',
    buttonId: 'vk_engine_gpu_btn'
  }
};

async function callGradioBridge(action, payload) {
  const bridge = BRIDGE_ACTIONS[action];
  if (!bridge) {
    throw new Error(`Unknown bridge action: ${action}`);
  }

  const inputField = getBridgeField(bridge.inputId);
  const outputField = getBridgeField(bridge.outputId);
  const button = getBridgeButton(bridge.buttonId);
  if (!inputField || !outputField || !button) {
    throw new Error(`Bridge controls missing for action: ${action}`);
  }

  const requestPayload = payload === undefined ? {} : payload;
  setFieldValue(inputField, JSON.stringify(requestPayload));

  const previousOutput = outputField.value || '';
  setFieldValue(outputField, '');
  button.click();

  const outputText = await waitForFieldUpdate(outputField, previousOutput);
  try {
    return JSON.parse(outputText);
  } catch (err) {
    throw new Error(`Invalid bridge JSON for ${action}: ${outputText}`);
  }
}

function sortEventsChronologically(eventsToSort) {
  return [...eventsToSort].sort((a, b) => {
    const ta = Number(a.time) || 0;
    const tb = Number(b.time) || 0;
    if (ta !== tb) return ta - tb;
    if (a.type === b.type) return 0;
    if (a.type === 'note_off') return -1;
    if (b.type === 'note_off') return 1;
    return 0;
  });
}

function sanitizeEvents(rawEvents) {
  if (!Array.isArray(rawEvents) || rawEvents.length === 0) {
    return [];
  }

  const cleaned = rawEvents
    .filter(e => e && (e.type === 'note_on' || e.type === 'note_off'))
    .map(e => ({
      type: e.type,
      note: Number(e.note) || 0,
      velocity: Number(e.velocity) || 0,
      time: Number(e.time) || 0,
      channel: Number(e.channel) || 0
    }));

  if (cleaned.length === 0) {
    return [];
  }

  return sortEventsChronologically(cleaned);
}

function normalizeEventsToZero(rawEvents) {
  const cleaned = sanitizeEvents(rawEvents);
  if (cleaned.length === 0) {
    return [];
  }

  const minTime = Math.min(...cleaned.map(e => e.time));
  return sortEventsChronologically(
    cleaned.map(e => ({
      ...e,
      time: Math.max(0, e.time - minTime)
    }))
  );
}

function clampMidiNote(note) {
  const minNote = baseMidi;
  const maxNote = baseMidi + (numOctaves * 12) - 1;
  return Math.max(minNote, Math.min(maxNote, note));
}

function eventsToNotePairs(rawEvents) {
  const pairs = [];
  const activeByNote = new Map();
  const sorted = sortEventsChronologically(rawEvents);

  sorted.forEach(event => {
    const note = Number(event.note) || 0;
    const time = Number(event.time) || 0;
    const velocity = Number(event.velocity) || 100;

    if (event.type === 'note_on' && velocity > 0) {
      if (!activeByNote.has(note)) activeByNote.set(note, []);
      activeByNote.get(note).push({ start: time, velocity });
      return;
    }

    if (event.type === 'note_off' || (event.type === 'note_on' && velocity <= 0)) {
      const stack = activeByNote.get(note);
      if (!stack || stack.length === 0) return;
      const active = stack.shift();
      const end = Math.max(active.start + 0.05, time);
      pairs.push({
        note: clampMidiNote(note),
        start: active.start,
        end,
        velocity: Math.max(1, Math.min(127, active.velocity))
      });
    }
  });

  return pairs.sort((a, b) => a.start - b.start);
}

function notePairsToEvents(pairs) {
  const eventsOut = [];
  pairs.forEach(pair => {
    const note = clampMidiNote(Math.round(pair.note));
    const start = Math.max(0, Number(pair.start) || 0);
    const end = Math.max(start + 0.05, Number(pair.end) || start + 0.2);
    const velocity = Math.max(1, Math.min(127, Math.round(Number(pair.velocity) || 100)));

    eventsOut.push({
      type: 'note_on',
      note,
      velocity,
      time: start,
      channel: 0
    });
    eventsOut.push({
      type: 'note_off',
      note,
      velocity: 0,
      time: end,
      channel: 0
    });
  });
  return sortEventsChronologically(eventsOut);
}

function trimNotePairs(pairs, maxNotes, maxDurationSec) {
  const out = [];
  for (let i = 0; i < pairs.length; i++) {
    if (out.length >= maxNotes) break;
    if (pairs[i].start > maxDurationSec) break;
    const boundedEnd = Math.min(pairs[i].end, maxDurationSec);
    out.push({
      ...pairs[i],
      end: Math.max(pairs[i].start + 0.05, boundedEnd)
    });
  }
  return out;
}

function smoothPairLeaps(pairs, maxLeapSemitones = 7) {
  if (pairs.length <= 1) return pairs;
  const smoothed = [{ ...pairs[0], note: clampMidiNote(pairs[0].note) }];
  for (let i = 1; i < pairs.length; i++) {
    const prev = smoothed[i - 1].note;
    let current = pairs[i].note;
    while (Math.abs(current - prev) > maxLeapSemitones) {
      current += current > prev ? -12 : 12;
    }
    smoothed.push({
      ...pairs[i],
      note: clampMidiNote(current)
    });
  }
  return smoothed;
}

function appendMotifEcho(pairs, callEvents, maxDurationSec) {
  const callPitches = normalizeEventsToZero(callEvents)
    .filter(e => e.type === 'note_on' && e.velocity > 0)
    .map(e => clampMidiNote(Number(e.note) || 0))
    .slice(0, 2);

  if (callPitches.length === 0) return pairs;

  let nextStart = pairs.length > 0 ? pairs[pairs.length - 1].end + 0.1 : 0.2;
  const out = [...pairs];
  callPitches.forEach((pitch, idx) => {
    const start = nextStart + (idx * 0.28);
    if (start >= maxDurationSec) return;
    out.push({
      note: pitch,
      start,
      end: Math.min(maxDurationSec, start + 0.22),
      velocity: 96
    });
  });
  return out;
}

function applyPlayfulShift(pairs) {
  return pairs.map((pair, idx) => {
    if (idx % 2 === 0) return pair;
    const direction = idx % 4 === 1 ? 2 : -2;
    return {
      ...pair,
      note: clampMidiNote(pair.note + direction)
    };
  });
}

// Generic preset getter - consolidates 3 similar functions
function getSelectedPreset(selectElement, presetMap, defaultKey, idKey) {
  const id = selectElement ? selectElement.value : defaultKey;
  return {
    [idKey]: id,
    ...(presetMap[id] || presetMap[defaultKey])
  };
}

function getSelectedStylePreset() {
  return getSelectedPreset(responseStyleSelect, RESPONSE_STYLE_PRESETS, 'melodic', 'styleId');
}

function getSelectedResponseMode() {
  return getSelectedPreset(responseModeSelect, RESPONSE_MODES, 'raw_godzilla', 'modeId');
}

function getSelectedResponseLengthPreset() {
  return getSelectedPreset(responseLengthSelect, RESPONSE_LENGTH_PRESETS, 'short', 'lengthId');
}

function getSelectedGameQuantization() {
  const modeId = quantizationSelect ? quantizationSelect.value : 'sixteenth';
  return {
    modeId,
    ...(GAME_QUANTIZATION_PRESETS[modeId] || GAME_QUANTIZATION_PRESETS.sixteenth)
  };
}

function getSelectedGameBars(selectElement, fallback = 2) {
  const raw = selectElement ? Number(selectElement.value) : fallback;
  if (raw === 1 || raw === 2) return raw;
  return fallback;
}

function getSelectedUserBars() {
  return getSelectedGameBars(userBarsSelect, 2);
}

function getSelectedAIBars() {
  return getSelectedGameBars(aiBarsSelect, 2);
}

function getDecodingOptionsForMode(modeId) {
  if (modeId === 'raw_godzilla') {
    return { temperature: 1.0, top_p: 0.98, num_candidates: 1 };
  }
  if (modeId === 'musical_polish') {
    return { temperature: 0.85, top_p: 0.93, num_candidates: 4 };
  }
  return { temperature: 0.9, top_p: 0.95, num_candidates: 3 };
}

function getSelectedDecodingOptions() {
  const mode = getSelectedResponseMode();
  return getDecodingOptionsForMode(mode.modeId);
}

function getSelectedRuntime() {
  if (!runtimeSelect || !runtimeSelect.value) return 'auto';
  return runtimeSelect.value;
}

function getRuntimeModeLabel(mode) {
  if (mode === 'gpu') return 'ZeroGPU';
  if (mode === 'auto') return 'Auto (GPU->CPU)';
  return 'CPU';
}

function resetAutoRuntimeResolution() {
  resolvedAutoRuntimeMode = null;
}

function resolveAutoRuntimeMode(engineId) {
  if (engineId !== 'godzilla_continue') {
    return 'cpu';
  }

  if (resolvedAutoRuntimeMode === 'cpu' || resolvedAutoRuntimeMode === 'gpu') {
    return resolvedAutoRuntimeMode;
  }

  const runtimeInfo = serverConfig && typeof serverConfig === 'object'
    ? serverConfig.runtime
    : null;
  const defaultMode = runtimeInfo && typeof runtimeInfo.default_mode === 'string'
    ? runtimeInfo.default_mode
    : null;

  if (defaultMode === 'cpu' || defaultMode === 'gpu') {
    resolvedAutoRuntimeMode = defaultMode;
  } else {
    resolvedAutoRuntimeMode = 'gpu';
  }

  const label = resolvedAutoRuntimeMode === 'gpu' ? 'ZeroGPU' : 'CPU';
  logToTerminal(`Runtime auto resolved to ${label} and will stay fixed this session.`, 'timestamp');
  return resolvedAutoRuntimeMode;
}

async function probeZeroGpuAvailabilityOnInit() {
  if (getSelectedRuntime() !== 'auto') return;
  if (autoRuntimeProbeInFlight) return;

  autoRuntimeProbeInFlight = true;
  try {
    logToTerminal('Runtime auto: probing ZeroGPU availability...', 'timestamp');
    const probePayload = {
      engine_id: 'parrot',
      events: [
        { type: 'note_on', note: 60, velocity: 64, time: 0, channel: 0 },
        { type: 'note_off', note: 60, velocity: 0, time: 0.1, channel: 0 }
      ],
      options: {}
    };
    const probeResult = await callGradioBridge('process_engine_gpu', probePayload);
    if (probeResult && !probeResult.error && Array.isArray(probeResult.events)) {
      resolvedAutoRuntimeMode = 'gpu';
      logToTerminal('Runtime auto probe: ZeroGPU available. Auto locked to ZeroGPU.', 'timestamp');
    } else {
      resolvedAutoRuntimeMode = 'cpu';
      const reason = probeResult && probeResult.error ? probeResult.error : 'unavailable';
      logToTerminal(`Runtime auto probe: ZeroGPU unavailable (${reason}). Auto locked to CPU.`, 'timestamp');
    }
  } catch (err) {
    resolvedAutoRuntimeMode = 'cpu';
    logToTerminal(`Runtime auto probe failed (${err.message}). Auto locked to CPU.`, 'timestamp');
  } finally {
    autoRuntimeProbeInFlight = false;
  }
}

function beatSec() {
  return 60 / GAME_BPM;
}

function barSec() {
  return beatSec() * GAME_BEATS_PER_BAR;
}

function barsToSeconds(bars) {
  return Math.max(1, Number(bars) || 1) * barSec();
}

function nowGameSec() {
  return nowSec() - gameClockOriginSec;
}

function nextBarAlignedStart(minLeadBeats = GAME_COUNTIN_BEATS) {
  const minStart = nowGameSec() + (Math.max(0, minLeadBeats) * beatSec());
  const barLength = barSec();
  return Math.ceil(minStart / barLength) * barLength;
}

function quantizeToStep(value, step) {
  if (!Number.isFinite(value) || !Number.isFinite(step) || step <= 0) {
    return value;
  }
  return Math.round(value / step) * step;
}

function moveByOctaveTowardTarget(note, target) {
  let candidate = note;
  while (candidate + 12 <= target) {
    candidate += 12;
  }
  while (candidate - 12 >= target) {
    candidate -= 12;
  }
  const up = clampMidiNote(candidate + 12);
  const down = clampMidiNote(candidate - 12);
  const current = clampMidiNote(candidate);
  const best = [current, up, down].reduce((winner, value) => {
    return Math.abs(value - target) < Math.abs(winner - target) ? value : winner;
  }, current);
  return clampMidiNote(best);
}

function getCallProfile(callEvents) {
  const normalizedCall = normalizeEventsToZero(callEvents);
  const pitches = normalizedCall
    .filter(e => e.type === 'note_on' && e.velocity > 0)
    .map(e => clampMidiNote(Number(e.note) || baseMidi));
  const velocities = normalizedCall
    .filter(e => e.type === 'note_on' && e.velocity > 0)
    .map(e => Math.max(1, Math.min(127, Number(e.velocity) || 100)));

  const keyboardCenter = baseMidi + Math.floor((numOctaves * 12) / 2);
  const center = pitches.length > 0
    ? pitches.reduce((sum, value) => sum + value, 0) / pitches.length
    : keyboardCenter;
  const finalPitch = pitches.length > 0 ? pitches[pitches.length - 1] : keyboardCenter;
  const avgVelocity = velocities.length > 0
    ? velocities.reduce((sum, value) => sum + value, 0) / velocities.length
    : 100;

  return { pitches, center, finalPitch, avgVelocity };
}

function applyResponseStyle(rawResponseEvents, callEvents, lengthPreset) {
  const preset = getSelectedStylePreset();
  const targetMaxNotes = Math.max(preset.maxNotes, lengthPreset.maxNotes);
  const targetMaxDuration = Math.max(preset.maxDurationSec, lengthPreset.maxDurationSec);
  let notePairs = eventsToNotePairs(normalizeEventsToZero(rawResponseEvents));
  notePairs = trimNotePairs(notePairs, targetMaxNotes, targetMaxDuration);
  if (preset.playfulShift) {
    notePairs = applyPlayfulShift(notePairs);
  }
  if (preset.smoothLeaps) {
    notePairs = smoothPairLeaps(notePairs);
  }
  if (preset.addMotifEcho) {
    notePairs = appendMotifEcho(notePairs, callEvents, targetMaxDuration);
    notePairs = trimNotePairs(notePairs, targetMaxNotes, targetMaxDuration);
  }
  return {
    styleLabel: preset.label,
    events: notePairsToEvents(notePairs)
  };
}

function applyMusicalPolish(rawResponseEvents, callEvents, lengthPreset) {
  const stylePreset = getSelectedStylePreset();
  const callProfile = getCallProfile(callEvents);
  let notePairs = eventsToNotePairs(normalizeEventsToZero(rawResponseEvents));

  if (notePairs.length === 0) {
    const fallbackPitches = callProfile.pitches.slice(0, 4);
    if (fallbackPitches.length === 0) {
      return [];
    }
    notePairs = fallbackPitches.map((pitch, idx) => {
      const start = idx * 0.28;
      return {
        note: clampMidiNote(pitch),
        start,
        end: start + 0.24,
        velocity: Math.round(callProfile.avgVelocity)
      };
    });
  }

  const polished = [];
  let previousStart = -1;
  for (let i = 0; i < notePairs.length; i++) {
    const source = notePairs[i];
    let note = moveByOctaveTowardTarget(source.note, callProfile.center);
    if (polished.length > 0) {
      const prev = polished[polished.length - 1].note;
      while (Math.abs(note - prev) > 7) {
        note += note > prev ? -12 : 12;
      }
      note = clampMidiNote(note);
    }

    const quantizedStart = Math.max(0, quantizeToStep(source.start, 0.125));
    const start = Math.max(quantizedStart, previousStart + 0.06);
    previousStart = start;

    const rawDur = Math.max(0.1, source.end - source.start);
    const duration = Math.max(0.12, Math.min(0.9, quantizeToStep(rawDur, 0.0625)));
    const velocity = Math.round(
      (Math.max(1, Math.min(127, source.velocity)) * 0.6)
      + (callProfile.avgVelocity * 0.4)
    );

    polished.push({
      note,
      start,
      end: start + duration,
      velocity: Math.max(1, Math.min(127, velocity))
    });
  }

  if (polished.length > 0) {
    polished[polished.length - 1].note = moveByOctaveTowardTarget(
      polished[polished.length - 1].note,
      callProfile.finalPitch
    );
  }

  let out = trimNotePairs(polished, lengthPreset.maxNotes, lengthPreset.maxDurationSec);
  if (stylePreset.addMotifEcho) {
    out = appendMotifEcho(out, callEvents, lengthPreset.maxDurationSec);
  }
  if (stylePreset.playfulShift) {
    out = applyPlayfulShift(out);
  }
  out = smoothPairLeaps(out, 6);
  out = trimNotePairs(out, lengthPreset.maxNotes, lengthPreset.maxDurationSec);
  return out;
}

function buildProcessedAIResponse(rawResponseEvents, callEvents) {
  const mode = getSelectedResponseMode();
  const lengthPreset = getSelectedResponseLengthPreset();

  if (mode.modeId === 'raw_godzilla') {
    return {
      label: `${mode.label} (${lengthPreset.label})`,
      events: normalizeEventsToZero(rawResponseEvents || [])
    };
  }

  if (mode.modeId === 'musical_polish') {
    return {
      label: `${mode.label} (${lengthPreset.label})`,
      events: notePairsToEvents(applyMusicalPolish(rawResponseEvents || [], callEvents, lengthPreset))
    };
  }

  const styled = applyResponseStyle(rawResponseEvents || [], callEvents, lengthPreset);
  return {
    label: `${mode.label} / ${styled.styleLabel} (${lengthPreset.label})`,
    events: styled.events
  };
}

function buildGameProcessedAIResponse(rawResponseEvents, callEvents, aiBars) {
  const mode = getSelectedResponseMode();
  const gameLengthPreset = {
    label: `${aiBars} bar${aiBars > 1 ? 's' : ''}`,
    maxNotes: aiBars === 1 ? 12 : 24,
    maxDurationSec: barsToSeconds(aiBars)
  };

  if (mode.modeId === 'raw_godzilla') {
    return {
      label: `${mode.label} (${gameLengthPreset.label})`,
      events: normalizeEventsToZero(rawResponseEvents || [])
    };
  }

  if (mode.modeId === 'musical_polish') {
    return {
      label: `${mode.label} (${gameLengthPreset.label})`,
      events: notePairsToEvents(applyMusicalPolish(rawResponseEvents || [], callEvents, gameLengthPreset))
    };
  }

  const styled = applyResponseStyle(rawResponseEvents || [], callEvents, gameLengthPreset);
  return {
    label: `${mode.label} / ${styled.styleLabel} (${gameLengthPreset.label})`,
    events: styled.events
  };
}

function clampValue(value, minValue, maxValue) {
  return Math.max(minValue, Math.min(maxValue, value));
}

function stretchNotePairsToDuration(pairs, targetDurationSec) {
  if (!Array.isArray(pairs) || pairs.length === 0) {
    return [];
  }

  const safeTarget = Math.max(0.1, Number(targetDurationSec) || 0.1);
  const sourceEnd = pairs.reduce((maxEnd, pair) => Math.max(maxEnd, Number(pair.end) || 0), 0);

  if (sourceEnd <= 0) {
    const spacing = safeTarget / Math.max(1, pairs.length);
    return pairs.map((pair, idx) => {
      const start = idx * spacing;
      const end = Math.min(safeTarget, start + Math.max(0.08, spacing * 0.8));
      return {
        ...pair,
        start,
        end: Math.max(start + 0.08, end)
      };
    });
  }

  const scale = safeTarget / sourceEnd;
  return pairs.map((pair) => ({
    ...pair,
    start: Math.max(0, (Number(pair.start) || 0) * scale),
    end: Math.max(0, (Number(pair.end) || 0) * scale)
  }));
}

function quantizeAiResponseForGame(rawEvents, aiBars) {
  const maxDurationSec = barsToSeconds(aiBars);
  const quantPreset = getSelectedGameQuantization();
  const quantStepSec = Number.isFinite(quantPreset.stepBeats)
    ? beatSec() * quantPreset.stepBeats
    : null;
  const minDurationSec = quantStepSec
    ? Math.max(0.08, quantStepSec * 0.5)
    : 0.08;

  const rawPairs = eventsToNotePairs(normalizeEventsToZero(rawEvents));
  if (rawPairs.length === 0) {
    return [];
  }
  const pairs = stretchNotePairsToDuration(rawPairs, maxDurationSec);

  const out = [];
  pairs.forEach((pair) => {
    const rawStart = quantStepSec ? quantizeToStep(pair.start, quantStepSec) : pair.start;
    const quantizedStart = clampValue(
      rawStart,
      0,
      Math.max(0, maxDurationSec - minDurationSec)
    );
    const rawEnd = quantStepSec ? quantizeToStep(pair.end, quantStepSec) : pair.end;
    const end = clampValue(
      Math.max(quantizedStart + minDurationSec, rawEnd),
      quantizedStart + minDurationSec,
      maxDurationSec
    );

    if (end - quantizedStart < minDurationSec * 0.75) {
      return;
    }

    out.push({
      note: clampMidiNote(Math.round(pair.note)),
      start: quantizedStart,
      end,
      velocity: clampValue(Math.round(pair.velocity || 100), 1, 127)
    });
  });

  return notePairsToEvents(out);
}

function getGameGenerateTokens() {
  return 32;
}

function getGridPhaseLabel(phase) {
  const labels = {
    idle: 'Idle',
    starting: 'Starting',
    user_countdown: 'User Count-In',
    user_turn: 'User Turn',
    ai_thinking: 'AI Thinking',
    ai_countdown: 'AI Count-In',
    ai_playback: 'AI Playback'
  };
  return labels[phase] || 'Idle';
}

function prepareCanvasContext(canvas) {
  if (!canvas) return null;
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return ctx;
}

function getGridPairs(eventsInput, maxDurationSec) {
  const pairs = eventsToNotePairs(sanitizeEvents(eventsInput));
  const maxDur = Math.max(0.1, Number(maxDurationSec) || 0.1);
  return pairs
    .filter(pair => pair.start < maxDur)
    .map(pair => ({
      ...pair,
      start: Math.max(0, pair.start),
      end: Math.max(pair.start + 0.05, Math.min(pair.end, maxDur))
    }));
}

function shouldAnimateGrid() {
  return gridPlayheads.user.active || gridPlayheads.ai.active;
}

function stopGridAnimationLoop() {
  if (gridAnimationFrameId !== null) {
    window.cancelAnimationFrame(gridAnimationFrameId);
    gridAnimationFrameId = null;
  }
}

function stopGridPlayhead(lane) {
  if (!gridPlayheads[lane]) return;
  gridPlayheads[lane].active = false;
  if (!shouldAnimateGrid()) {
    stopGridAnimationLoop();
  }
}

function stopAllGridPlayheads() {
  stopGridPlayhead('user');
  stopGridPlayhead('ai');
}

function readGridPlayheadSec(lane) {
  const state = gridPlayheads[lane];
  if (!state || !state.active) return null;

  const elapsed = Math.max(0, nowSec() - state.startWallSec);
  if (elapsed >= state.durationSec) {
    state.active = false;
    return state.durationSec;
  }
  return elapsed;
}

function runGridAnimationFrame() {
  if (!shouldAnimateGrid()) {
    gridAnimationFrameId = null;
    return;
  }
  renderTurnGrid({ phase: gamePhase });
  gridAnimationFrameId = window.requestAnimationFrame(runGridAnimationFrame);
}

function ensureGridAnimationLoop() {
  if (gridAnimationFrameId !== null) return;
  gridAnimationFrameId = window.requestAnimationFrame(runGridAnimationFrame);
}

function startGridPlayhead(lane, durationSec) {
  if (!gridPlayheads[lane]) return;
  gridPlayheads[lane].active = true;
  gridPlayheads[lane].startWallSec = nowSec();
  gridPlayheads[lane].durationSec = Math.max(0.1, Number(durationSec) || 0.1);
  ensureGridAnimationLoop();
}

function getEventsDurationSec(eventsInput) {
  if (!Array.isArray(eventsInput) || eventsInput.length === 0) {
    return 0.1;
  }
  return Math.max(
    0.1,
    ...eventsInput.map(event => Math.max(0, Number(event.time) || 0))
  );
}

function drawTurnGridLane(canvas, eventsInput, bars, laneType = 'user', playheadSec = null) {
  const ctx = prepareCanvasContext(canvas);
  if (!ctx) return;

  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  const padX = 8;
  const padY = 8;
  const innerW = Math.max(1, width - (padX * 2));
  const innerH = Math.max(1, height - (padY * 2));
  const rows = numOctaves * 12;
  const totalBars = Math.max(1, bars);
  const totalSixteenths = totalBars * 16;
  const maxDurationSec = barsToSeconds(totalBars);
  const rowH = innerH / rows;

  const bgGrad = ctx.createLinearGradient(0, 0, 0, height);
  if (laneType === 'ai') {
    bgGrad.addColorStop(0, 'rgba(31, 10, 48, 0.95)');
    bgGrad.addColorStop(1, 'rgba(8, 4, 18, 0.98)');
  } else {
    bgGrad.addColorStop(0, 'rgba(8, 16, 42, 0.95)');
    bgGrad.addColorStop(1, 'rgba(5, 5, 18, 0.98)');
  }
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = bgGrad;
  ctx.fillRect(0, 0, width, height);

  for (let row = 0; row <= rows; row++) {
    const y = padY + (row * rowH);
    ctx.strokeStyle = row % 12 === 0
      ? 'rgba(62, 244, 255, 0.2)'
      : 'rgba(62, 244, 255, 0.07)';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padX, y);
    ctx.lineTo(padX + innerW, y);
    ctx.stroke();
  }

  for (let step = 0; step <= totalSixteenths; step++) {
    const x = padX + ((step / totalSixteenths) * innerW);
    if (step % 16 === 0) {
      ctx.strokeStyle = laneType === 'ai'
        ? 'rgba(255, 63, 176, 0.58)'
        : 'rgba(62, 244, 255, 0.58)';
      ctx.lineWidth = 1.4;
    } else if (step % 4 === 0) {
      ctx.strokeStyle = 'rgba(154, 184, 255, 0.34)';
      ctx.lineWidth = 1.1;
    } else {
      ctx.strokeStyle = 'rgba(154, 184, 255, 0.12)';
      ctx.lineWidth = 1;
    }
    ctx.beginPath();
    ctx.moveTo(x, padY);
    ctx.lineTo(x, padY + innerH);
    ctx.stroke();
  }

  const minNote = baseMidi;
  const maxNote = baseMidi + (numOctaves * 12) - 1;
  const noteRange = Math.max(1, maxNote - minNote + 1);
  const pairs = getGridPairs(eventsInput, maxDurationSec);

  pairs.forEach((pair) => {
    const start = Math.max(0, Math.min(maxDurationSec, pair.start));
    const end = Math.max(start + 0.01, Math.min(maxDurationSec, pair.end));
    const x = padX + ((start / maxDurationSec) * innerW);
    const w = Math.max(2.5, ((end - start) / maxDurationSec) * innerW);

    const clampedNote = clampMidiNote(Math.round(pair.note));
    const noteRow = maxNote - clampedNote;
    const y = padY + (noteRow / noteRange) * innerH;
    const h = Math.max(3, rowH - 1.5);

    const fill = ctx.createLinearGradient(x, y, x + w, y + h);
    if (laneType === 'ai') {
      fill.addColorStop(0, 'rgba(255, 95, 196, 0.98)');
      fill.addColorStop(1, 'rgba(199, 92, 255, 0.92)');
      ctx.shadowColor = 'rgba(255, 63, 176, 0.55)';
    } else {
      fill.addColorStop(0, 'rgba(76, 255, 255, 0.96)');
      fill.addColorStop(1, 'rgba(76, 161, 255, 0.9)');
      ctx.shadowColor = 'rgba(62, 244, 255, 0.6)';
    }
    ctx.shadowBlur = 8;
    ctx.fillStyle = fill;
    ctx.fillRect(x, y + 0.6, w, h);
    ctx.shadowBlur = 0;

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.45)';
    ctx.lineWidth = 0.8;
    ctx.strokeRect(x, y + 0.6, w, h);
  });

  ctx.strokeStyle = 'rgba(198, 216, 255, 0.32)';
  ctx.lineWidth = 1;
  ctx.strokeRect(padX, padY, innerW, innerH);

  if (playheadSec !== null) {
    const clampedPlayhead = clampValue(playheadSec, 0, maxDurationSec);
    const x = padX + ((clampedPlayhead / maxDurationSec) * innerW);
    ctx.strokeStyle = laneType === 'ai'
      ? 'rgba(255, 95, 196, 0.95)'
      : 'rgba(76, 255, 255, 0.95)';
    ctx.lineWidth = 2;
    ctx.shadowBlur = 10;
    ctx.shadowColor = laneType === 'ai'
      ? 'rgba(255, 63, 176, 0.8)'
      : 'rgba(62, 244, 255, 0.85)';
    ctx.beginPath();
    ctx.moveTo(x, padY);
    ctx.lineTo(x, padY + innerH);
    ctx.stroke();
    ctx.shadowBlur = 0;
  }
}

function renderTurnGrid({
  userEvents = null,
  aiEvents = null,
  phase = gamePhase
} = {}) {
  if (userEvents !== null) {
    gameGridUserEvents = sanitizeEvents(userEvents);
  }
  if (aiEvents !== null) {
    gameGridAIEvents = sanitizeEvents(aiEvents);
  }

  const userBars = getSelectedUserBars();
  const aiBars = getSelectedAIBars();
  const userPlayheadSec = readGridPlayheadSec('user');
  const aiPlayheadSec = readGridPlayheadSec('ai');
  drawTurnGridLane(userGridCanvas, gameGridUserEvents, userBars, 'user', userPlayheadSec);
  drawTurnGridLane(aiGridCanvas, gameGridAIEvents, aiBars, 'ai', aiPlayheadSec);

  if (userGridMeta) {
    userGridMeta.textContent = `${userBars} bar${userBars > 1 ? 's' : ''}`;
  }
  if (aiGridMeta) {
    aiGridMeta.textContent = `${aiBars} bar${aiBars > 1 ? 's' : ''}`;
  }
  if (gridPhaseBadge) {
    gridPhaseBadge.textContent = getGridPhaseLabel(phase);
  }
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
  const captureTimestamp = recording
    ? (nowSec() - startTime)
    : (gameCaptureActive ? (nowSec() - gameCaptureStartWallSec) : null);
  const timestamp = captureTimestamp !== null ? captureTimestamp.toFixed(3) : '--';
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

  if (gameCaptureActive) {
    gameCaptureActiveNotes.add(midiNote);
    gameCapturedEvents.push({
      type: 'note_on',
      note: midiNote,
      velocity: Math.max(1, velocity | 0),
      time: Math.max(0, nowSec() - gameCaptureStartWallSec),
      channel: 0
    });
    renderTurnGrid({
      userEvents: getLiveGameCaptureEvents(),
      phase: gamePhase
    });
  }
}

function noteOff(midiNote) {
  const freq = Tone.Frequency(midiNote, "midi").toFrequency();
  synth.triggerRelease(freq);
  
  const noteName = midiToNoteName(midiNote);
  const captureTimestamp = recording
    ? (nowSec() - startTime)
    : (gameCaptureActive ? (nowSec() - gameCaptureStartWallSec) : null);
  const timestamp = captureTimestamp !== null ? captureTimestamp.toFixed(3) : '--';
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

  if (gameCaptureActive) {
    gameCaptureActiveNotes.delete(midiNote);
    gameCapturedEvents.push({
      type: 'note_off',
      note: midiNote,
      velocity: 0,
      time: Math.max(0, nowSec() - gameCaptureStartWallSec),
      channel: 0
    });
    renderTurnGrid({
      userEvents: getLiveGameCaptureEvents(),
      phase: gamePhase
    });
  }
}

// =============================================================================
// COMPUTER KEYBOARD INPUT
// =============================================================================

function getKeyElement(midiNote) {
  return keyboardEl.querySelector(`.key[data-midi="${midiNote}"]`);
}

function bindGlobalKeyboardHandlers() {
  document.addEventListener('keydown', async (ev) => {
    if (!keyboardToggle || !keyboardToggle.checked) return;
    const key = ev.key.toLowerCase();
    const activeKeyMap = window.keyMapFromServer || keyMap; // Use server config if available, fallback to hardcoded
    if (!activeKeyMap[key] || pressedKeys.has(key)) return;
    
    ev.preventDefault();
    pressedKeys.add(key);
    
    await Tone.start();
    
    const midiNote = activeKeyMap[key];
    const keyEl = getKeyElement(midiNote);
    if (keyEl) keyEl.style.filter = 'brightness(0.85)';
    noteOn(midiNote, 100);
  });

  document.addEventListener('keyup', (ev) => {
    if (!keyboardToggle || !keyboardToggle.checked) return;
    const key = ev.key.toLowerCase();
    const activeKeyMap = window.keyMapFromServer || keyMap; // Use server config if available, fallback to hardcoded
    if (!activeKeyMap[key] || !pressedKeys.has(key)) return;
    
    ev.preventDefault();
    pressedKeys.delete(key);
    
    const midiNote = activeKeyMap[key];
    const keyEl = getKeyElement(midiNote);
    if (keyEl) keyEl.style.filter = '';
    noteOff(midiNote);
  });
}

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
    const payload = await callGradioBridge('save_midi', events);
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

function clearTrackedGameTimeouts() {
  gameTimeoutIds.forEach((id) => clearTimeout(id));
  gameTimeoutIds.clear();
}

function scheduleGameTimeout(callback, delayMs) {
  const safeDelayMs = Math.max(0, Number(delayMs) || 0);
  const timeoutId = setTimeout(() => {
    gameTimeoutIds.delete(timeoutId);
    callback();
  }, safeDelayMs);
  gameTimeoutIds.add(timeoutId);
  return timeoutId;
}

function scheduleGameAt(targetGameSec, callback) {
  const delayMs = (targetGameSec - nowGameSec()) * 1000;
  return scheduleGameTimeout(callback, delayMs);
}

function hideCountdownOverlay() {
  if (!countdownOverlay || !countdownText) return;
  countdownOverlay.classList.remove('active');
  countdownText.classList.remove('pulse', 'go');
  countdownText.textContent = '';
}

function showCountdownCue(text, isGo = false) {
  if (!countdownOverlay || !countdownText) return;
  countdownOverlay.classList.add('active');
  countdownText.textContent = text;
  countdownText.classList.remove('pulse', 'go');
  // Force restart animation for each cue.
  // eslint-disable-next-line no-unused-expressions
  countdownText.offsetWidth;
  countdownText.classList.add('pulse');
  if (isGo) {
    countdownText.classList.add('go');
  }
}

function scheduleCountdown(targetStartSec, label, sessionId) {
  [3, 2, 1].forEach((count) => {
    scheduleGameAt(targetStartSec - (count * beatSec()), () => {
      if (!gameActive || sessionId !== gameSessionId) return;
      statusEl.textContent = `${label} in ${count}...`;
      showCountdownCue(String(count));
    });
  });

  scheduleGameAt(targetStartSec, () => {
    if (!gameActive || sessionId !== gameSessionId) return;
    statusEl.textContent = `${label}: GO`;
    showCountdownCue('GO', true);
  });

  scheduleGameAt(targetStartSec + (beatSec() * 0.7), () => {
    if (!gameActive || sessionId !== gameSessionId) return;
    hideCountdownOverlay();
  });
}

function ensureMetronomeInstruments() {
  if (!metronomeKick) {
    metronomeKick = new Tone.MembraneSynth({
      pitchDecay: 0.03,
      octaves: 4,
      envelope: {
        attack: 0.001,
        decay: 0.22,
        sustain: 0
      }
    }).toDestination();
  }

  if (!metronomeSnare) {
    metronomeSnare = new Tone.NoiseSynth({
      noise: { type: 'white' },
      envelope: {
        attack: 0.001,
        decay: 0.16,
        sustain: 0
      }
    }).toDestination();
  }

  if (!metronomeHat) {
    metronomeHat = new Tone.MetalSynth({
      frequency: 270,
      envelope: {
        attack: 0.001,
        decay: 0.08,
        release: 0.02
      },
      harmonicity: 4.1,
      modulationIndex: 18,
      resonance: 1200
    }).toDestination();
  }
}

function playMetronomeBeat(beatIndex) {
  const beatInBar = beatIndex % GAME_BEATS_PER_BAR;

  if (metronomeHat) {
    metronomeHat.triggerAttackRelease('16n', undefined, beatInBar === 0 ? 0.24 : 0.16);
  }
  if (metronomeKick && (beatInBar === 0 || beatInBar === 2)) {
    metronomeKick.triggerAttackRelease('C1', '8n', undefined, beatInBar === 0 ? 0.62 : 0.5);
  }
  if (metronomeSnare && (beatInBar === 1 || beatInBar === 3)) {
    metronomeSnare.triggerAttackRelease('16n', undefined, 0.28);
  }
}

function scheduleNextMetronomeBeat(sessionId) {
  if (!gameActive || sessionId !== gameSessionId) return;
  const targetBeatIndex = metronomeBeatIndex;
  const beatTimeSec = targetBeatIndex * beatSec();
  scheduleGameAt(beatTimeSec, () => {
    if (!gameActive || sessionId !== gameSessionId) return;
    playMetronomeBeat(targetBeatIndex);
    metronomeBeatIndex = targetBeatIndex + 1;
    scheduleNextMetronomeBeat(sessionId);
  });
}

function startGameMetronome(sessionId) {
  ensureMetronomeInstruments();
  metronomeBeatIndex = Math.max(0, Math.floor(nowGameSec() / beatSec()));
  scheduleNextMetronomeBeat(sessionId);
}

function stopGameMetronome() {
  // Beat scheduling is canceled via clearTrackedGameTimeouts().
  // Drum voices are one-shots so no sustained release handling is needed here.
}

async function processEventsThroughEngine(inputEvents, options = {}) {
  const selectedEngineId = engineSelect.value;
  if (!selectedEngineId || selectedEngineId === 'parrot') {
    return { events: inputEvents };
  }

  const requestStartedMs = performance.now();
  const requestOptions = { ...options };
  const runtimeMode = getSelectedRuntime();
  const effectiveRuntimeMode = runtimeMode === 'auto'
    ? resolveAutoRuntimeMode(selectedEngineId)
    : runtimeMode;
  if (
    selectedEngineId === 'godzilla_continue'
    && typeof requestOptions.generate_tokens !== 'number'
  ) {
    requestOptions.generate_tokens = RESPONSE_LENGTH_PRESETS.medium.generateTokens;
  }

  let bridgeAction = 'process_engine_cpu';
  let runtimeUsed = 'cpu';
  if (selectedEngineId === 'godzilla_continue') {
    if (effectiveRuntimeMode === 'gpu') {
      bridgeAction = 'process_engine_gpu';
      runtimeUsed = 'gpu';
    }
  }

  const requestPayload = {
    engine_id: selectedEngineId,
    events: inputEvents,
    options: requestOptions
  };

  let result;
  const primaryAttemptStartMs = performance.now();
  try {
    result = await callGradioBridge(bridgeAction, requestPayload);
  } catch (err) {
    const primaryAttemptMs = performance.now() - primaryAttemptStartMs;
    if (
      selectedEngineId === 'godzilla_continue'
      && runtimeMode === 'auto'
      && bridgeAction === 'process_engine_gpu'
    ) {
      logToTerminal(
        `Runtime auto: ZeroGPU failed after ${Math.round(primaryAttemptMs)}ms (${err.message}), retrying on CPU.`,
        'timestamp'
      );
      resolvedAutoRuntimeMode = 'cpu';
      logToTerminal('Runtime auto switched to CPU and will remain on CPU.', 'timestamp');
      const cpuAttemptStartMs = performance.now();
      result = await callGradioBridge('process_engine_cpu', requestPayload);
      runtimeUsed = 'cpu';
      const cpuAttemptMs = performance.now() - cpuAttemptStartMs;
      const totalMs = performance.now() - requestStartedMs;
      logToTerminal(
        `Inference fallback success on CPU in ${Math.round(cpuAttemptMs)}ms (total ${Math.round(totalMs)}ms).`,
        'timestamp'
      );
    } else {
      throw err;
    }
  }

  if (
    result
    && result.error
    && selectedEngineId === 'godzilla_continue'
    && runtimeMode === 'auto'
    && bridgeAction === 'process_engine_gpu'
  ) {
    const primaryAttemptMs = performance.now() - primaryAttemptStartMs;
    logToTerminal(
      `Runtime auto: ZeroGPU error after ${Math.round(primaryAttemptMs)}ms (${result.error}), retrying on CPU.`,
      'timestamp'
    );
    resolvedAutoRuntimeMode = 'cpu';
    logToTerminal('Runtime auto switched to CPU and will remain on CPU.', 'timestamp');
    const cpuAttemptStartMs = performance.now();
    result = await callGradioBridge('process_engine_cpu', requestPayload);
    runtimeUsed = 'cpu';
    const cpuAttemptMs = performance.now() - cpuAttemptStartMs;
    const totalMs = performance.now() - requestStartedMs;
    logToTerminal(
      `Inference fallback success on CPU in ${Math.round(cpuAttemptMs)}ms (total ${Math.round(totalMs)}ms).`,
      'timestamp'
    );
  }

  if (result && result.error) {
    throw new Error(result.error);
  }
  if (!result || !Array.isArray(result.events)) {
    throw new Error('Engine returned no events');
  }
  if (result.warning) {
    logToTerminal(`ENGINE WARNING: ${result.warning}`, 'timestamp');
  }

  if (selectedEngineId === 'godzilla_continue') {
    const totalMs = performance.now() - requestStartedMs;
    const tokens = Number(requestOptions.generate_tokens) || 0;
    const inCount = Array.isArray(inputEvents) ? inputEvents.length : 0;
    const outCount = Array.isArray(result.events) ? result.events.length : 0;
    logToTerminal(
      `Inference ${getRuntimeModeLabel(runtimeUsed)}: ${Math.round(totalMs)}ms | in=${inCount} ev | out=${outCount} ev | tokens=${tokens}`,
      'timestamp'
    );
  }

  return result;
}

function playEvents(
  eventsToPlay,
  { logSymbols = true, useAISynth = false, shouldAbort = null } = {}
) {
  return new Promise((resolve) => {
    if (!Array.isArray(eventsToPlay) || eventsToPlay.length === 0) {
      resolve();
      return;
    }

    const playbackSynth = useAISynth && aiSynth ? aiSynth : synth;
    const abortRequested = () => typeof shouldAbort === 'function' && shouldAbort();
    let finished = false;
    let eventIndex = 0;

    const finishPlayback = () => {
      if (finished) return;
      finished = true;
      if (playbackSynth) playbackSynth.releaseAll();
      keyboardEl.querySelectorAll('.key').forEach(k => {
        k.style.filter = '';
      });
      resolve();
    };

    const playEvent = () => {
      if (abortRequested()) {
        finishPlayback();
        return;
      }

      if (eventIndex >= eventsToPlay.length) {
        finishPlayback();
        return;
      }

      const event = eventsToPlay[eventIndex];
      const nextTime = eventIndex + 1 < eventsToPlay.length
        ? eventsToPlay[eventIndex + 1].time
        : event.time;

      if (event.type === 'note_on') {
        const freq = Tone.Frequency(event.note, "midi").toFrequency();
        if (playbackSynth) {
          playbackSynth.triggerAttack(freq, undefined, event.velocity / 127);
        }
        if (logSymbols) {
          const noteName = midiToNoteName(event.note);
          logToTerminal(
            `[${event.time.toFixed(3)}s] ► ${noteName} (${event.note})`,
            'note-on'
          );
        }
        const keyEl = getKeyElement(event.note);
        if (keyEl) keyEl.style.filter = 'brightness(0.7)';
      } else if (event.type === 'note_off') {
        const freq = Tone.Frequency(event.note, "midi").toFrequency();
        if (playbackSynth) {
          playbackSynth.triggerRelease(freq);
        }
        if (logSymbols) {
          const noteName = midiToNoteName(event.note);
          logToTerminal(
            `[${event.time.toFixed(3)}s] ◄ ${noteName}`,
            'note-off'
          );
        }
        const keyEl = getKeyElement(event.note);
        if (keyEl) keyEl.style.filter = '';
      }

      eventIndex++;
      const deltaTime = Math.max(0, nextTime - event.time);
      setTimeout(playEvent, deltaTime * 1000);
    };

    playEvent();
  });
}

async function startGameLoop() {
  if (gameActive) return;
  await Tone.start();

  if (recording) {
    stopRecord();
  }

  gameSessionId += 1;
  const sessionId = gameSessionId;
  gameActive = true;
  gamePhase = 'starting';
  gameCaptureActive = false;
  gameCapturedEvents = [];
  gameCaptureActiveNotes.clear();
  gameGridUserEvents = [];
  gameGridAIEvents = [];
  stopAllGridPlayheads();
  hideCountdownOverlay();
  clearTrackedGameTimeouts();
  gameClockOriginSec = nowSec();
  gameTurn = 0;
  gameStartBtn.disabled = true;
  gameStopBtn.disabled = false;
  recordBtn.disabled = true;
  stopBtn.disabled = true;
  playbackBtn.disabled = true;
  saveBtn.disabled = true;
  statusEl.textContent = 'Game started';

  logToTerminal('', '');
  logToTerminal('🎮 CALL & RESPONSE GAME STARTED', 'timestamp');
  const quantizationPreset = getSelectedGameQuantization();
  logToTerminal(
    `Tempo locked at ${GAME_BPM} BPM, beat grid 4/4, AI quantize: ${quantizationPreset.label}.`,
    'timestamp'
  );
  logToTerminal(
    `Bars: user=${getSelectedUserBars()} | ai=${getSelectedAIBars()} (adjust anytime).`,
    'timestamp'
  );
  const stylePreset = getSelectedStylePreset();
  const modePreset = getSelectedResponseMode();
  const decodingPreset = getSelectedDecodingOptions();
  logToTerminal(
    `AI mode: ${modePreset.label} | style: ${stylePreset.label}`,
    'timestamp'
  );
  logToTerminal(
    `Decoding: temp=${decodingPreset.temperature} top_p=${decodingPreset.top_p} candidates=${decodingPreset.num_candidates}`,
    'timestamp'
  );
  logToTerminal('', '');
  renderTurnGrid({ phase: gamePhase });

  startGameMetronome(sessionId);
  scheduleUserTurn(sessionId);
}

function stopGameLoop(reason = 'Game stopped') {
  gameSessionId += 1;
  clearTrackedGameTimeouts();
  hideCountdownOverlay();
  stopGameMetronome();
  stopAllGridPlayheads();
  gamePhase = 'idle';
  gameCaptureActive = false;
  gameCapturedEvents = [];
  gameCaptureActiveNotes.clear();
  if (recording) {
    stopRecord();
  }
  gameActive = false;
  gameStartBtn.disabled = false;
  gameStopBtn.disabled = true;
  recordBtn.disabled = false;
  stopBtn.disabled = true;
  playbackBtn.disabled = events.length === 0;
  saveBtn.disabled = events.length === 0;
  statusEl.textContent = reason;
  if (synth) synth.releaseAll();
  if (aiSynth) aiSynth.releaseAll();
  keyboardEl.querySelectorAll('.key').forEach(k => {
    k.style.filter = '';
  });
  renderTurnGrid({ phase: gamePhase });
  logToTerminal(`🎮 ${reason}`, 'timestamp');
}

function beginUserCaptureWindow(sessionId, userBars) {
  if (!gameActive || sessionId !== gameSessionId) return;
  gamePhase = 'user_turn';
  gameCaptureActive = true;
  gameCaptureStartWallSec = nowSec();
  gameCapturedEvents = [];
  gameCaptureActiveNotes.clear();
  gameGridUserEvents = [];
  startGridPlayhead('user', barsToSeconds(userBars));
  stopGridPlayhead('ai');
  statusEl.textContent = `Turn ${gameTurn}: your call (${userBars} bar${userBars > 1 ? 's' : ''})`;
  renderTurnGrid({
    userEvents: [],
    phase: gamePhase
  });
  logToTerminal(`Turn ${gameTurn}: your call started`, 'timestamp');
}

function finalizeOpenGameCaptureNotes(captureDurationSec) {
  if (gameCaptureActiveNotes.size === 0) return;
  const closeTime = Math.max(0, captureDurationSec);
  gameCaptureActiveNotes.forEach((note) => {
    gameCapturedEvents.push({
      type: 'note_off',
      note,
      velocity: 0,
      time: closeTime,
      channel: 0
    });
  });
  gameCaptureActiveNotes.clear();
}

function getLiveGameCaptureEvents() {
  const live = [...gameCapturedEvents];
  if (!gameCaptureActive) {
    return sortEventsChronologically(live);
  }

  const nowCaptureSec = Math.max(0, nowSec() - gameCaptureStartWallSec);
  gameCaptureActiveNotes.forEach((note) => {
    live.push({
      type: 'note_off',
      note,
      velocity: 0,
      time: nowCaptureSec,
      channel: 0
    });
  });
  return sortEventsChronologically(live);
}

function scheduleUserTurn(sessionId) {
  if (!gameActive || sessionId !== gameSessionId) return;
  gameTurn += 1;
  const userBars = getSelectedUserBars();
  const userStartSec = nextBarAlignedStart(GAME_COUNTIN_BEATS);
  const userEndSec = userStartSec + barsToSeconds(userBars);

  gamePhase = 'user_countdown';
  gameGridUserEvents = [];
  gameGridAIEvents = [];
  stopAllGridPlayheads();
  renderTurnGrid({
    userEvents: [],
    aiEvents: [],
    phase: gamePhase
  });
  logToTerminal(
    `Turn ${gameTurn}: user countdown (${userBars} bar${userBars > 1 ? 's' : ''})`,
    'timestamp'
  );
  scheduleCountdown(userStartSec, `Turn ${gameTurn}: your turn`, sessionId);

  scheduleGameAt(userStartSec, () => beginUserCaptureWindow(sessionId, userBars));
  scheduleGameAt(userEndSec, () => {
    void finishUserTurn(sessionId);
  });
}

async function finishUserTurn(sessionId) {
  if (!gameActive || sessionId !== gameSessionId) return;
  stopGridPlayhead('user');
  const captureDurationSec = Math.max(0, nowSec() - gameCaptureStartWallSec);
  finalizeOpenGameCaptureNotes(captureDurationSec);
  gameCaptureActive = false;
  const userGridEvents = sanitizeEvents(gameCapturedEvents);
  const callEvents = normalizeEventsToZero(gameCapturedEvents);
  gameCapturedEvents = [];
  renderTurnGrid({
    userEvents: userGridEvents,
    phase: gamePhase
  });

  if (callEvents.length === 0) {
    statusEl.textContent = `Turn ${gameTurn}: no notes, try again`;
    logToTerminal('No notes captured, restarting your turn...', 'timestamp');
    scheduleGameTimeout(() => {
      scheduleUserTurn(sessionId);
    }, GAME_RETRY_DELAY_MS);
    return;
  }

  try {
    gamePhase = 'ai_thinking';
    renderTurnGrid({ phase: gamePhase });
    statusEl.textContent = `Turn ${gameTurn}: AI thinking...`;
    logToTerminal(`Turn ${gameTurn}: AI is thinking...`, 'timestamp');

    const aiBars = getSelectedAIBars();
    const decodingOptions = getSelectedDecodingOptions();
    const result = await processEventsThroughEngine(callEvents, {
      generate_tokens: getGameGenerateTokens(),
      ...decodingOptions
    });
    const processedResponse = buildGameProcessedAIResponse(result.events || [], callEvents, aiBars);
    const aiEvents = quantizeAiResponseForGame(processedResponse.events, aiBars);

    if (!gameActive || sessionId !== gameSessionId) return;
    if (aiEvents.length === 0) {
      logToTerminal('AI returned no playable events after quantization. Restarting turn.', 'timestamp');
      scheduleUserTurn(sessionId);
      return;
    }

    const aiStartSec = nextBarAlignedStart(GAME_COUNTIN_BEATS);
    gamePhase = 'ai_countdown';
    renderTurnGrid({
      aiEvents,
      phase: gamePhase
    });
    logToTerminal(
      `Turn ${gameTurn}: AI countdown (${aiBars} bar${aiBars > 1 ? 's' : ''})`,
      'timestamp'
    );
    scheduleCountdown(aiStartSec, `Turn ${gameTurn}: AI`, sessionId);

    scheduleGameAt(aiStartSec, async () => {
      if (!gameActive || sessionId !== gameSessionId) return;
      gamePhase = 'ai_playback';
      startGridPlayhead('ai', getEventsDurationSec(aiEvents));
      renderTurnGrid({ phase: gamePhase });
      statusEl.textContent = `Turn ${gameTurn}: AI responds`;
      logToTerminal(
        `Turn ${gameTurn}: AI response (${processedResponse.label}, ${aiBars} bar${aiBars > 1 ? 's' : ''})`,
        'timestamp'
      );
      await playEvents(aiEvents, {
        useAISynth: true,
        shouldAbort: () => !gameActive || sessionId !== gameSessionId
      });
      stopGridPlayhead('ai');

      if (!gameActive || sessionId !== gameSessionId) return;
      scheduleUserTurn(sessionId);
    });

    if (!gameActive || sessionId !== gameSessionId) return;
    logToTerminal(
      `Turn ${gameTurn}: AI ready (${processedResponse.label})`,
      'timestamp'
    );
  } catch (err) {
    console.error('Game turn error:', err);
    logToTerminal(`ENGINE ERROR: ${err.message}`, 'timestamp');
    stopGameLoop(`Game stopped: ${err.message}`);
  }
}

// =============================================================================
// HELPER FUNCTIONS
// =============================================================================

// Reset all audio synthesis and visual key states
function resetAllNotesAndVisuals() {
  if (synth) synth.releaseAll();
  if (aiSynth) aiSynth.releaseAll();
  keyboardEl.querySelectorAll('.key').forEach(k => {
    k.style.filter = '';
  });
}

// =============================================================================
// EVENT LISTENERS
// =============================================================================

let listenersBound = false;

function bindUIEventListeners() {
  if (listenersBound) return;
  listenersBound = true;

  bindGlobalKeyboardHandlers();

  if (instrumentSelect) {
    instrumentSelect.addEventListener('change', () => {
      loadInstrument(instrumentSelect.value);
    });
  }

  if (aiInstrumentSelect) {
    aiInstrumentSelect.addEventListener('change', () => {
      loadAIInstrument(aiInstrumentSelect.value);
      logToTerminal(`AI voice switched to: ${aiInstrumentSelect.value}`, 'timestamp');
    });
  }

  if (keyboardToggle) {
    keyboardToggle.addEventListener('change', () => {
      if (keyboardToggle.checked) {
        // Show keyboard shortcuts
        keyboardEl.classList.add('shortcuts-visible');
      } else {
        // Hide keyboard shortcuts
        keyboardEl.classList.remove('shortcuts-visible');
        // Release all currently pressed keyboard keys
        pressedKeys.forEach(key => {
          const activeKeyMap = window.keyMapFromServer || keyMap;
          const midiNote = activeKeyMap[key];
          const keyEl = getKeyElement(midiNote);
          if (keyEl) keyEl.style.filter = '';
          noteOff(midiNote);
        });
        pressedKeys.clear();
      }
    });
  }

  if (clearTerminal) {
    clearTerminal.addEventListener('click', () => {
      terminal.innerHTML = '';
      logToTerminal('╔═══════════════════════════════════════════════════════╗', 'timestamp');
      logToTerminal('║         🎹 MIDI MONITOR INITIALIZED 🎹              ║', 'timestamp');
      logToTerminal('╚═══════════════════════════════════════════════════════╝', 'timestamp');
      logToTerminal('Ready to capture MIDI events...', 'timestamp');
      logToTerminal('', '');
    });
  }

  if (recordBtn) {
    recordBtn.addEventListener('click', async () => {
      if (gameActive) return;
      await Tone.start();
      beginRecord();
    });
  }

  if (stopBtn) {
    stopBtn.addEventListener('click', () => {
      if (gameActive) return;
      stopRecord();
    });
  }

  if (engineSelect) {
    engineSelect.addEventListener('change', (e) => {
      selectedEngine = e.target.value;
      logToTerminal(`Engine switched to: ${selectedEngine}`, 'timestamp');
    });
  }

  // Consolidated select control listeners
  const selectControls = [
    {
      element: runtimeSelect,
      getter: () => {
        const mode = getSelectedRuntime();
        const label = mode === 'gpu' ? 'ZeroGPU' : (mode === 'auto' ? 'Auto (GPU->CPU)' : 'CPU');
        return { label };
      },
      message: (result) => `Runtime switched to: ${result.label}`
    },
    {
      element: responseStyleSelect,
      getter: getSelectedStylePreset,
      message: (result) => `AI style switched to: ${result.label}`
    },
    {
      element: responseModeSelect,
      getter: () => {
        const mode = getSelectedResponseMode();
        const decode = getSelectedDecodingOptions();
        return { label: `${mode.label} (temp=${decode.temperature}, top_p=${decode.top_p}, candidates=${decode.num_candidates})` };
      },
      message: (result) => `Response mode switched to: ${result.label}`
    },
    {
      element: responseLengthSelect,
      getter: () => {
        const preset = getSelectedResponseLengthPreset();
        return { label: `${preset.label} (${preset.generateTokens} tokens)` };
      },
      message: (result) => (
        gameActive
          ? `Response length switched to: ${result.label} (game mode uses bar controls)`
          : `Response length switched to: ${result.label}`
      )
    },
    {
      element: quantizationSelect,
      getter: getSelectedGameQuantization,
      message: (result) => `Game quantization switched to: ${result.label}`
    },
    {
      element: userBarsSelect,
      getter: () => {
        const bars = getSelectedUserBars();
        return { label: `${bars} bar${bars > 1 ? 's' : ''}` };
      },
      message: (result) => `Game user bars switched to: ${result.label}`
    },
    {
      element: aiBarsSelect,
      getter: () => {
        const bars = getSelectedAIBars();
        return { label: `${bars} bar${bars > 1 ? 's' : ''}` };
      },
      message: (result) => `Game AI bars switched to: ${result.label}`
    }
  ];

  selectControls.forEach(({ element, getter, message }) => {
    if (element) {
      element.addEventListener('change', () => {
        const result = getter();
        if (element === runtimeSelect && getSelectedRuntime() === 'auto') {
          resetAutoRuntimeResolution();
          void probeZeroGpuAvailabilityOnInit();
        }
        logToTerminal(message(result), 'timestamp');
        if (element === userBarsSelect || element === aiBarsSelect) {
          renderTurnGrid({ phase: gamePhase });
        }
      });
    }
  });

  // Setup hover tooltips for control items
  const setupControlItemHoverListeners = () => {
    const controlIdToName = {
      'engine': 'Engine',
      'runtime': 'Runtime',
      'aiStyle': 'AI Style',
      'responseMode': 'Response Mode',
      'responseLength': 'Response Length',
      'quantization': 'AI Quantization',
      'userBars': 'User Bars',
      'aiBars': 'AI Bars',
      'instrument': 'Instrument',
      'aiVoice': 'AI Voice'
    };
    
    const updateTooltipDisplay = (html) => {
      const tooltipDisplay = document.getElementById('tooltipDisplay');
      if (tooltipDisplay) {
        tooltipDisplay.innerHTML = html;
      }
    };
    
    const controlItems = document.querySelectorAll('[data-control-id]');
    controlItems.forEach(label => {
      const select = label.querySelector('select');
      if (!select) return;
      
      const controlId = label.getAttribute('data-control-id');
      const controlName = controlIdToName[controlId] || controlId;
      const description = select.getAttribute('data-description');
      const showOption = ['engine', 'runtime', 'aiStyle', 'responseMode', 'responseLength', 'quantization', 'userBars', 'aiBars'].includes(controlId);
      
      const updateDisplay = () => {
        if (showOption) {
          const selectedOption = select.querySelector(`option[value="${select.value}"]`);
          if (selectedOption) {
            const optionName = selectedOption.textContent;
            const tooltip = selectedOption.getAttribute('data-tooltip');
            if (tooltip && description) {
              updateTooltipDisplay(`<b>${controlName}</b>: ${description} - <b>${optionName}</b>: ${tooltip}`);
            } else if (description) {
              updateTooltipDisplay(`<b>${controlName}</b>: ${description}`);
            }
          }
        } else if (description) {
          updateTooltipDisplay(`<b>${controlName}</b>: ${description}`);
        }
      };
      
      label.addEventListener('mouseenter', updateDisplay);
      label.addEventListener('mouseleave', () => {
        updateTooltipDisplay('');
      });
    });
  };
  setupControlItemHoverListeners();

  if (gameStartBtn) {
    gameStartBtn.addEventListener('click', () => {
      void startGameLoop();
    });
  }

  if (gameStopBtn) {
    gameStopBtn.addEventListener('click', () => {
      stopGameLoop('Game stopped');
    });
  }

  if (playbackBtn) {
    playbackBtn.addEventListener('click', async () => {
      if (gameActive) return alert('Stop the game first.');
      if (events.length === 0) return alert('No recording to play back');
      
      // Ensure all notes are off before starting playback
      resetAllNotesAndVisuals();
      
      statusEl.textContent = 'Playing back...';
      playbackBtn.disabled = true;
      recordBtn.disabled = true;
      
      logToTerminal('', '');
      logToTerminal('♫♫♫ PLAYBACK STARTED ♫♫♫', 'timestamp');
      logToTerminal('', '');
      
      try {
        let engineOptions = {};
        if (engineSelect.value === 'godzilla_continue') {
          const lengthPreset = getSelectedResponseLengthPreset();
          engineOptions = {
            generate_tokens: lengthPreset.generateTokens,
            ...getSelectedDecodingOptions()
          };
        }
        const result = await processEventsThroughEngine(events, engineOptions);
        let processedEvents = result.events || [];
        if (engineSelect.value === 'godzilla_continue') {
          const processedResponse = buildProcessedAIResponse(processedEvents, events);
          processedEvents = processedResponse.events;
          logToTerminal(`Playback response mode: ${processedResponse.label}`, 'timestamp');
        }
        await playEvents(processedEvents, {
          useAISynth: engineSelect.value !== 'parrot'
        });

        statusEl.textContent = 'Playback complete';
        playbackBtn.disabled = false;
        recordBtn.disabled = false;
        logToTerminal('', '');
        logToTerminal('♫♫♫ PLAYBACK FINISHED ♫♫♫', 'timestamp');
        logToTerminal('', '');
      } catch (err) {
        console.error('Playback error:', err);
        statusEl.textContent = 'Playback error: ' + err.message;
        logToTerminal(`ENGINE ERROR: ${err.message}`, 'timestamp');
        playbackBtn.disabled = false;
        recordBtn.disabled = false;
        
        // Ensure all notes are off on error
        resetAllNotesAndVisuals();
      }
    });
  }

  if (saveBtn) {
    saveBtn.addEventListener('click', () => saveMIDI());
  }

  if (panicBtn) {
    panicBtn.addEventListener('click', () => {
      // Stop all notes immediately and reset visuals
      resetAllNotesAndVisuals();
      
      // Clear all pressed keys
      pressedKeys.clear();
      
      logToTerminal('🚨 PANIC - All notes stopped', 'timestamp');
    });
  }

  window.addEventListener('resize', () => {
    renderTurnGrid({ phase: gamePhase });
  });
}

// =============================================================================
// =============================================================================
// INITIALIZATION
// =============================================================================

async function init() {
  await waitForKeyboardUIElements();
  await waitForBridgeElements();
  cacheUIElements();
  bindUIEventListeners();

  // First, load configuration from server
  await initializeFromConfig();

  if (responseStyleSelect && !responseStyleSelect.value) {
    responseStyleSelect.value = 'melodic';
  }
  if (responseModeSelect && !responseModeSelect.value) {
    responseModeSelect.value = 'raw_godzilla';
  }
  if (responseLengthSelect && !responseLengthSelect.value) {
    responseLengthSelect.value = 'short';
  }
  if (quantizationSelect && !quantizationSelect.value) {
    quantizationSelect.value = 'sixteenth';
  }
  if (runtimeSelect && !runtimeSelect.value) {
    runtimeSelect.value = 'auto';
  }
  if (userBarsSelect && !userBarsSelect.value) {
    userBarsSelect.value = '2';
  }
  if (aiBarsSelect && !aiBarsSelect.value) {
    aiBarsSelect.value = '2';
  }
  if (aiInstrumentSelect && !aiInstrumentSelect.value) {
    aiInstrumentSelect.value = 'fm';
  }
  
  // Then load default instrument (synth)
  loadInstrument('synth');
  loadAIInstrument(aiInstrumentSelect ? aiInstrumentSelect.value : 'fm');
  
  // Setup keyboard event listeners and UI
  attachPointerEvents();
  initTerminal();
  renderTurnGrid({ phase: gamePhase });
  const runtimeMode = getSelectedRuntime();
  const runtimeLabel = runtimeMode === 'gpu' ? 'ZeroGPU' : (runtimeMode === 'auto' ? 'Auto (GPU->CPU)' : 'CPU');
  logToTerminal(`Runtime mode: ${runtimeLabel}`, 'timestamp');
  if (runtimeMode === 'auto') {
    logToTerminal('Runtime auto probe started in background...', 'timestamp');
    void probeZeroGpuAvailabilityOnInit();
  }
  logToTerminal(`Game mode tempo: ${GAME_BPM} BPM (fixed)`, 'timestamp');
  // Set initial button states
  recordBtn.disabled = false;
  stopBtn.disabled = true;
  saveBtn.disabled = true;
  playbackBtn.disabled = true;
  gameStartBtn.disabled = false;
  gameStopBtn.disabled = true;
}

// Start the application when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => {
    void init();
  });
} else {
  void init();
}
