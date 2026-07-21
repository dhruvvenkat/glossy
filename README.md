# Glossy

Hold Right Alt, ask a question, then release it. Glossy records the hold with
`arecord`, transcribes it locally with Whisper, asks the configured OpenAI model,
and reads the answer with a local Piper neural voice. Holds shorter than the
configured threshold are ignored; a blip starts recording and a reversed blip
marks its end. While recording, the live transcript refreshes in the terminal and a
small voice-reactive indicator appears at the bottom-center of the screen.

While Glossy is speaking, press Escape to stop playback and discard that
question and answer. Holding the configured push-to-talk button also discards
it, then starts a new recording once the normal hold threshold is reached.

## Setup

```sh
python3 -m venv venv
venv/bin/pip install -r requirements.txt
venv/bin/python -m piper.download_voices --download-dir voices en_US-lessac-medium
sudo usermod -aG input "$USER"
```

Log out and back in after adding the group. Membership in `input` allows
processes running as your user to read keyboard events, so only run trusted
software under your account.

Glossy reads this existing config file:

```sh
# ~/.config/glossy.env
OPENAI_API_KEY=your-key
```

Keep it private with `chmod 600 ~/.config/glossy.env`.

Runtime behavior lives in `config/config.json`:

```json
{
  "model": "gpt-5.5",
  "reasoning_effort": "medium",
  "transcription_model": "small.en",
  "transcription_beam_size": 1,
  "hold_seconds": 1.0,
  "button": "KEY_RIGHTALT",
  "visualizer_sensitivity": 4.0,
  "speech_rms_threshold": 300,
  "minimum_speech_seconds": 0.3,
  "vad_aggressiveness": 3,
  "speech_snr_ratio": 2.0
}
```

Use Linux evdev key names for `button`. Set `reasoning_effort` to `null` for
models that do not support reasoning controls. Restart Glossy after editing the
file. `transcription_model` controls the local Whisper size; `small.en` is the
default accuracy/latency balance for English on CPU. Raise
`transcription_beam_size` for potentially better transcription at the cost of
latency. Use `base.en` or `tiny.en` if live terminal dictation lags. The model
downloads on first startup and then runs locally. Raise
`visualizer_sensitivity` if the recording bars move too little. Downloaded
models are kept in the repository's ignored `models/` directory.
Raise `speech_rms_threshold` if background noise is being uploaded; lower it if
quiet speech is skipped. Recordings need at least `minimum_speech_seconds` above
that threshold and classified as speech by the local WebRTC detector before
Glossy contacts OpenAI. `vad_aggressiveness` ranges from `0` to `3`; higher
values reject more non-speech noise. `speech_snr_ratio` requires speech to be
louder than the noise measured at the start of each recording.

Run Glossy in the foreground first:

```sh
venv/bin/python src/ask.py
```

Edit `config/system-prompt.md` to control answer length and style, then restart Glossy.

## Reading threads

Glossy can keep separate context for books, chapters, papers, or other material.
Use the same push-to-talk button and say one of these commands:

```text
new thread Operating Systems
switch to thread Operating Systems
list threads
current thread
threads mode
exit threads mode
```

Creating or switching a thread enables threads mode. While it is enabled, Glossy
uses the thread's six most recent questions and answers as context. After every
five answered questions, it also condenses the new conversation into a short
long-term memory. Exiting threads mode preserves the selected thread but returns
questions to the normal context-free behavior.

Saying `list threads` replaces the live transcript with a thread picker. Use Up
and Down to move, Enter to switch and hear confirmation, or Escape to cancel.
You can still switch directly by saying `switch to thread` followed by its name.
While the picker is open, Glossy captures the keyboard so those keys do not reach
the background app.

Threads are local JSON files in `~/.config/glossy/threads/`. Version 1 learns
from your conversation; it does not import the book, PDF, or web page itself.

## Code layout

- `src/ask.py` starts the application.
- `src/listener.py` owns keyboard events and the push-to-talk flow.
- `src/audio.py` owns recording, transcription, speech detection, and playback.
- `src/openai_api.py` owns answer and thread-summary API requests.
- `src/threads.py` owns thread commands, context, and local persistence.
- `src/settings.py` owns environment and runtime configuration validation.
- `src/visualizer.py` owns the on-screen recording indicator.

## Change voice

List the English Piper voices, type one of their names, and make it active with:

```sh
venv/bin/python src/choose_voice.py
```

The utility downloads the voice, selects it, and restarts Glossy automatically.

Glossy checks its keyboard connection every five seconds. After suspend or a
keyboard swap, it waits quietly and reconnects as soon as a compatible keyboard
appears; systemd remains the crash-recovery fallback.

## Start automatically

The included unit assumes this repository remains at
`~/Documents/Programming/glossy`.

```sh
mkdir -p ~/.config/systemd/user
ln -sf "$PWD/config/glossy.service" ~/.config/systemd/user/glossy.service
systemctl --user daemon-reload
systemctl --user enable --now glossy.service
```

View errors with `journalctl --user -u glossy.service -f`.

## Check

```sh
venv/bin/python -m unittest -v
```
