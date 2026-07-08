# Glossy

Hold Right Alt, ask a question, then release it. Glossy records the hold with
`arecord`, transcribes it with OpenAI Whisper, asks the configured OpenAI model,
and reads the answer with a local Piper neural voice. Holds shorter than the
configured threshold are ignored; a blip starts recording and a reversed blip
marks its end. While recording, a small voice-reactive indicator appears at the
bottom-center of the screen.

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

Runtime behavior lives in `config.json`:

```json
{
  "model": "gpt-5.5",
  "reasoning_effort": "medium",
  "hold_seconds": 1.0,
  "button": "KEY_RIGHTALT",
  "visualizer_sensitivity": 4.0,
  "speech_rms_threshold": 300,
  "minimum_speech_seconds": 0.15
}
```

Use Linux evdev key names for `button`. Set `reasoning_effort` to `null` for
models that do not support reasoning controls. Restart Glossy after editing the
file. Raise `visualizer_sensitivity` if the recording bars move too little.
Raise `speech_rms_threshold` if background noise is being uploaded; lower it if
quiet speech is skipped. Recordings need at least `minimum_speech_seconds` above
that threshold before Glossy contacts OpenAI.

Run Glossy in the foreground first:

```sh
venv/bin/python ask.py
```

Edit `system-prompt.md` to control answer length and style, then restart Glossy.

## Change voice

List the English Piper voices, type one of their names, and make it active with:

```sh
venv/bin/python choose_voice.py
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
ln -sf "$PWD/glossy.service" ~/.config/systemd/user/glossy.service
systemctl --user daemon-reload
systemctl --user enable --now glossy.service
```

View errors with `journalctl --user -u glossy.service -f`.

## Check

```sh
venv/bin/python -m unittest -v
```
