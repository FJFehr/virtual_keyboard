# Static Assets

This directory contains the client-side assets for the Virtual MIDI Keyboard.

## Files

- **styles.css** - All application styles including keyboard, controls, and MIDI terminal
- **keyboard.js** - Client-side logic for:
  - Keyboard rendering and layout
  - Audio synthesis (Tone.js integration)
  - MIDI event recording
  - Computer keyboard input handling
  - MIDI monitor/terminal
  - File export functionality

## Usage

These files are loaded by `keyboard.html` via `/file=` paths in Gradio.
