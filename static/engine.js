/**
 * Virtual MIDI Keyboard - Engine
 * 
 * The engine processes MIDI signals and can:
 * - Capture and store MIDI events
 * - Play back captured sequences
 * - Process and transform MIDI data
 * 
 * This is the core processing unit for all signal manipulation.
 */

// =============================================================================
// ENGINE STATE
// =============================================================================

class MIDIEngine {
  constructor() {
    this.recordings = {}; // Store multiple recordings
    this.currentRecordingId = null;
    this.isPlayingBack = false;
    this.playbackSpeed = 1.0; // 1.0 = normal speed
    this.callbacks = {
      onNoteOn: null,
      onNoteOff: null,
      onPlaybackStart: null,
      onPlaybackEnd: null,
      onProgress: null
    };
  }

  /**
   * Start recording a new performance
   * Returns the recording ID
   */
  startRecording(recordingName = 'Recording_' + Date.now()) {
    this.currentRecordingId = recordingName;
    this.recordings[recordingName] = {
      name: recordingName,
      events: [],
      startTime: performance.now(),
      duration: 0,
      metadata: {
        createdAt: new Date().toISOString(),
        instrument: 'default'
      }
    };
    return recordingName;
  }

  /**
   * Stop recording and return the recording
   */
  stopRecording() {
    if (!this.currentRecordingId) return null;

    const recording = this.recordings[this.currentRecordingId];
    if (recording && recording.events.length > 0) {
      const lastEvent = recording.events[recording.events.length - 1];
      recording.duration = lastEvent.time;
    }

    const recordingId = this.currentRecordingId;
    this.currentRecordingId = null;
    return recording;
  }

  /**
   * Add a MIDI event to the current recording
   */
  addEvent(event) {
    if (!this.currentRecordingId) return false;

    const recording = this.recordings[this.currentRecordingId];
    if (!recording) return false;

    recording.events.push({
      ...event,
      id: recording.events.length
    });

    return true;
  }

  /**
   * Get a recording by ID
   */
  getRecording(recordingId) {
    return this.recordings[recordingId];
  }

  /**
   * Get all recordings
   */
  getAllRecordings() {
    return Object.values(this.recordings);
  }

  /**
   * Delete a recording
   */
  deleteRecording(recordingId) {
    if (this.currentRecordingId === recordingId) {
      this.stopRecording();
    }
    delete this.recordings[recordingId];
  }

  /**
   * Play back a recording
   * Calls the provided callbacks for each note
   */
  async playback(recordingId, callbacks = {}) {
    const recording = this.getRecording(recordingId);
    if (!recording || recording.events.length === 0) {
      console.warn('No recording found or recording is empty');
      return;
    }

    // Merge provided callbacks with instance callbacks
    const finalCallbacks = { ...this.callbacks, ...callbacks };

    this.isPlayingBack = true;
    if (finalCallbacks.onPlaybackStart) {
      finalCallbacks.onPlaybackStart(recording);
    }

    const events = recording.events;
    let playbackStartTime = performance.now();

    for (let i = 0; i < events.length; i++) {
      if (!this.isPlayingBack) break; // Stop if playback was cancelled

      const event = events[i];
      const scheduledTime = (event.time / this.playbackSpeed) * 1000; // ms
      const now = performance.now() - playbackStartTime;
      const waitTime = scheduledTime - now;

      if (waitTime > 0) {
        await this.sleep(waitTime);
      }

      // Execute the event callback
      if (event.type === 'note_on' && finalCallbacks.onNoteOn) {
        finalCallbacks.onNoteOn(event);
      } else if (event.type === 'note_off' && finalCallbacks.onNoteOff) {
        finalCallbacks.onNoteOff(event);
      }

      // Progress callback
      if (finalCallbacks.onProgress) {
        finalCallbacks.onProgress({
          currentIndex: i,
          totalEvents: events.length,
          currentTime: event.time,
          totalDuration: recording.duration,
          progress: (i / events.length) * 100
        });
      }
    }

    this.isPlayingBack = false;
    if (finalCallbacks.onPlaybackEnd) {
      finalCallbacks.onPlaybackEnd(recording);
    }
  }

  /**
   * Stop playback
   */
  stopPlayback() {
    this.isPlayingBack = false;
  }

  /**
   * Set playback speed (1.0 = normal, 2.0 = double speed, etc.)
   */
  setPlaybackSpeed(speed) {
    this.playbackSpeed = Math.max(0.1, Math.min(2.0, speed));
  }

  /**
   * Export recording as JSON
   */
  exportAsJSON(recordingId) {
    const recording = this.getRecording(recordingId);
    if (!recording) return null;
    return JSON.stringify(recording, null, 2);
  }

  /**
   * Import recording from JSON
   */
  importFromJSON(jsonString, recordingName = null) {
    try {
      const recording = JSON.parse(jsonString);
      const id = recordingName || recording.name || 'imported_' + Date.now();
      this.recordings[id] = recording;
      return id;
    } catch (error) {
      console.error('Failed to import recording:', error);
      return null;
    }
  }

  /**
   * Get recording statistics
   */
  getStats(recordingId) {
    const recording = this.getRecording(recordingId);
    if (!recording) return null;

    const events = recording.events;
    const noteOnEvents = events.filter(e => e.type === 'note_on');
    const noteOffEvents = events.filter(e => e.type === 'note_off');

    return {
      recordingName: recording.name,
      totalEvents: events.length,
      noteOnCount: noteOnEvents.length,
      noteOffCount: noteOffEvents.length,
      duration: recording.duration,
      avgNotesPerSecond: (noteOnEvents.length / recording.duration).toFixed(2),
      minNote: Math.min(...noteOnEvents.map(e => e.note)),
      maxNote: Math.max(...noteOnEvents.map(e => e.note)),
      instrument: recording.metadata?.instrument || 'unknown'
    };
  }

  /**
   * Helper: Sleep for ms milliseconds
   */
  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// =============================================================================
// EXPORT
// =============================================================================

// Create a global engine instance
const midiEngine = new MIDIEngine();
