# Glossy

Hold Caps Lock, ask a question, then release it. Glossy records the hold with
`arecord`, transcribes it with OpenAI Whisper, asks the configured OpenAI model,
and reads the answer with a local Piper neural voice. Holds shorter than 350 ms
are ignored.

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
GLOSSY_MODEL=gpt-5.5
```

Keep it private with `chmod 600 ~/.config/glossy.env`. Run Glossy in the
foreground first:

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

On GNOME, disable Caps Lock's normal typing behavior in GNOME Tweaks if you do
not want each question to toggle capitalization. Glossy still receives the raw
physical key event.

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
