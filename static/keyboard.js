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
let terminal = null;
let clearTerminal = null;

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
let gameTurnTimerId = null;
let gameTurnTimeoutId = null;

const USER_TURN_LIMIT_SEC = 6;
const GAME_NEXT_TURN_DELAY_MS = 800;

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
  terminal = document.getElementById('terminal');
  clearTerminal = document.getElementById('clearTerminal');
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
    'terminal',
    'clearTerminal'
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

function normalizeEventsToZero(rawEvents) {
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

function clearGameTimers() {
  if (gameTurnTimerId !== null) {
    clearInterval(gameTurnTimerId);
    gameTurnTimerId = null;
  }
  if (gameTurnTimeoutId !== null) {
    clearTimeout(gameTurnTimeoutId);
    gameTurnTimeoutId = null;
  }
}

async function processEventsThroughEngine(inputEvents, options = {}) {
  const selectedEngineId = engineSelect.value;
  if (!selectedEngineId || selectedEngineId === 'parrot') {
    return { events: inputEvents };
  }

  const requestOptions = { ...options };
  const runtimeMode = getSelectedRuntime();
  if (
    selectedEngineId === 'godzilla_continue'
    && typeof requestOptions.generate_tokens !== 'number'
  ) {
    requestOptions.generate_tokens = RESPONSE_LENGTH_PRESETS.medium.generateTokens;
  }

  let bridgeAction = 'process_engine_cpu';
  if (selectedEngineId === 'godzilla_continue') {
    if (runtimeMode === 'gpu' || runtimeMode === 'auto') {
      bridgeAction = 'process_engine_gpu';
    }
  }

  const requestPayload = {
    engine_id: selectedEngineId,
    events: inputEvents,
    options: requestOptions
  };

  let result;
  try {
    result = await callGradioBridge(bridgeAction, requestPayload);
  } catch (err) {
    if (
      selectedEngineId === 'godzilla_continue'
      && runtimeMode === 'auto'
      && bridgeAction === 'process_engine_gpu'
    ) {
      logToTerminal('Runtime auto: ZeroGPU failed, retrying on CPU.', 'timestamp');
      result = await callGradioBridge('process_engine_cpu', requestPayload);
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
    logToTerminal(`Runtime auto: ZeroGPU error (${result.error}), retrying on CPU.`, 'timestamp');
    result = await callGradioBridge('process_engine_cpu', requestPayload);
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

  return result;
}

function playEvents(eventsToPlay, { logSymbols = true, useAISynth = false } = {}) {
  return new Promise((resolve) => {
    if (!Array.isArray(eventsToPlay) || eventsToPlay.length === 0) {
      resolve();
      return;
    }

    const playbackSynth = useAISynth && aiSynth ? aiSynth : synth;
    let eventIndex = 0;

    const playEvent = () => {
      if (eventIndex >= eventsToPlay.length) {
        if (playbackSynth) playbackSynth.releaseAll();
        keyboardEl.querySelectorAll('.key').forEach(k => {
          k.style.filter = '';
        });
        resolve();
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

  if (engineSelect.querySelector('option[value="godzilla_continue"]')) {
    engineSelect.value = 'godzilla_continue';
    selectedEngine = 'godzilla_continue';
  }

  gameActive = true;
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
  logToTerminal(
    `Flow: ${USER_TURN_LIMIT_SEC}s call, AI response, repeat until you stop.`,
    'timestamp'
  );
  const stylePreset = getSelectedStylePreset();
  const modePreset = getSelectedResponseMode();
  const lengthPreset = getSelectedResponseLengthPreset();
  const decodingPreset = getSelectedDecodingOptions();
  logToTerminal(
    `AI mode: ${modePreset.label} | length: ${lengthPreset.label} | style: ${stylePreset.label}`,
    'timestamp'
  );
  logToTerminal(
    `Decoding: temp=${decodingPreset.temperature} top_p=${decodingPreset.top_p} candidates=${decodingPreset.num_candidates}`,
    'timestamp'
  );
  logToTerminal('', '');

  await startUserTurn();
}

function stopGameLoop(reason = 'Game stopped') {
  clearGameTimers();
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
  logToTerminal(`🎮 ${reason}`, 'timestamp');
}

async function startUserTurn() {
  if (!gameActive) return;
  clearGameTimers();

  gameTurn += 1;
  beginRecord();
  gameStartBtn.disabled = true;
  gameStopBtn.disabled = false;
  recordBtn.disabled = true;
  stopBtn.disabled = true;
  playbackBtn.disabled = true;
  saveBtn.disabled = true;

  let remaining = USER_TURN_LIMIT_SEC;
  statusEl.textContent = `Turn ${gameTurn}: your call (${remaining}s)`;
  logToTerminal(`Turn ${gameTurn}: your call starts now`, 'timestamp');

  gameTurnTimerId = setInterval(() => {
    if (!gameActive) return;
    remaining -= 1;
    if (remaining > 0) {
      statusEl.textContent = `Turn ${gameTurn}: your call (${remaining}s)`;
    }
  }, 1000);

  gameTurnTimeoutId = setTimeout(() => {
    void finishUserTurn();
  }, USER_TURN_LIMIT_SEC * 1000);
}

async function finishUserTurn() {
  if (!gameActive) return;
  clearGameTimers();
  if (recording) stopRecord();
  recordBtn.disabled = true;
  stopBtn.disabled = true;
  playbackBtn.disabled = true;
  saveBtn.disabled = true;

  const callEvents = [...events];
  if (callEvents.length === 0) {
    statusEl.textContent = `Turn ${gameTurn}: no notes, try again`;
    logToTerminal('No notes captured, restarting your turn...', 'timestamp');
    setTimeout(() => {
      void startUserTurn();
    }, GAME_NEXT_TURN_DELAY_MS);
    return;
  }

  try {
    statusEl.textContent = `Turn ${gameTurn}: AI thinking...`;
    logToTerminal(`Turn ${gameTurn}: AI is thinking...`, 'timestamp');

    const lengthPreset = getSelectedResponseLengthPreset();
    const promptEvents = normalizeEventsToZero(callEvents);
    const decodingOptions = getSelectedDecodingOptions();
    const result = await processEventsThroughEngine(promptEvents, {
      generate_tokens: lengthPreset.generateTokens,
      ...decodingOptions
    });
    const processedResponse = buildProcessedAIResponse(result.events || [], callEvents);
    const aiEvents = processedResponse.events;

    if (!gameActive) return;

    statusEl.textContent = `Turn ${gameTurn}: AI responds`;
    logToTerminal(
      `Turn ${gameTurn}: AI response (${processedResponse.label})`,
      'timestamp'
    );
    await playEvents(aiEvents, { useAISynth: true });

    if (!gameActive) return;
    setTimeout(() => {
      void startUserTurn();
    }, GAME_NEXT_TURN_DELAY_MS);
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
      message: (result) => `Response length switched to: ${result.label}`
    }
  ];

  selectControls.forEach(({ element, getter, message }) => {
    if (element) {
      element.addEventListener('change', () => {
        const result = getter();
        logToTerminal(message(result), 'timestamp');
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
      const showOption = ['engine', 'runtime', 'aiStyle', 'responseMode', 'responseLength'].includes(controlId);
      
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
  if (runtimeSelect && !runtimeSelect.value) {
    runtimeSelect.value = 'auto';
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
  const runtimeMode = getSelectedRuntime();
  const runtimeLabel = runtimeMode === 'gpu' ? 'ZeroGPU' : (runtimeMode === 'auto' ? 'Auto (GPU->CPU)' : 'CPU');
  logToTerminal(`Runtime mode: ${runtimeLabel}`, 'timestamp');
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
