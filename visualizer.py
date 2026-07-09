#!/usr/bin/env python3

import math
import re
import subprocess
import sys
import tkinter as tk
from array import array
from pathlib import Path

WIDTH = 88
HEIGHT = 32
QUESTION_WIDTH = 560
BACKGROUND = "#111827"
BAR_COLOR = "#60a5fa"
TEXT_COLOR = "#f9fafb"
BAR_WIDTH = 4
BAR_SPACING = 9
BAR_DROP_DELAY = 0.08
TEXT_PAD_X = 16
TEXT_TOP = 7
BAR_BOTTOM_PAD = 12


def primary_geometry(default_width, default_height):
    try:
        result = subprocess.run(
            ["xrandr", "--listactivemonitors"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return 0, 0, default_width, default_height

    for line in result.stdout.splitlines():
        if "*" not in line:
            continue
        match = re.search(r"(\d+)/\d+x(\d+)/\d+([+-]\d+)([+-]\d+)", line)
        if match:
            width, height, x, y = map(int, match.groups())
            return x, y, width, height
    return 0, 0, default_width, default_height


def bottom_center(root, width, height):
    screen_x, screen_y, screen_width, screen_height = primary_geometry(
        root.winfo_screenwidth(), root.winfo_screenheight()
    )
    x = screen_x + (screen_width - width) // 2
    y = screen_y + screen_height - height - 48
    return f"{width}x{height}+{x}+{y}"


def audio_level(path, sensitivity=1.0):
    try:
        size = path.stat().st_size
        with path.open("rb") as audio:
            audio.seek(max(44, size - 3200))
            data = audio.read()
    except OSError:
        return 0.0

    data = data[: len(data) // 2 * 2]
    if not data:
        return 0.0
    samples = array("h")
    samples.frombytes(data)
    if sys.byteorder != "little":
        samples.byteswap()
    rms = math.sqrt(sum(sample * sample for sample in samples) / len(samples))
    return min(1.0, rms / 6000 * sensitivity)


def eased(value):
    value = max(0.0, min(1.0, value))
    return 1 - (1 - value) ** 3


def cascade_progress(progress, index):
    delay = index * BAR_DROP_DELAY
    if progress <= delay:
        return 0.0
    return eased((progress - delay) / (1 - delay))


def transcript_size(text_box, screen_width):
    if not text_box:
        return WIDTH, HEIGHT
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]
    max_width = max(220, min(QUESTION_WIDTH, screen_width - 64))
    return (
        max(WIDTH, min(max_width, text_width + TEXT_PAD_X * 2)),
        max(HEIGHT, text_height + 38),
    )


def main(audio_path, sensitivity, question_path=None):
    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.attributes("-alpha", 0.92)
    root.geometry(bottom_center(root, WIDTH, HEIGHT))

    canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg=BACKGROUND, highlightthickness=0)
    canvas.pack()
    text = canvas.create_text(
        WIDTH // 2,
        TEXT_TOP,
        text="",
        fill=TEXT_COLOR,
        font=("Sans", 11),
        justify=tk.CENTER,
        state=tk.HIDDEN,
        anchor=tk.N,
    )
    bars = [
        canvas.create_line(0, 0, 0, 0, fill=BAR_COLOR, width=BAR_WIDTH, capstyle=tk.ROUND)
        for _ in range(5)
    ]
    smoothed = 0.0
    phase = 0.0
    drop = 0.0

    def animate():
        nonlocal smoothed, phase, drop
        transcript = ""
        if question_path is not None:
            try:
                transcript = question_path.read_text().strip()
            except OSError:
                pass
        drop = min(1.0, drop + 0.08) if transcript else max(0.0, drop - 0.12)

        screen_width = root.winfo_screenwidth()
        max_width = max(220, min(QUESTION_WIDTH, screen_width - 64))
        canvas.itemconfigure(
            text,
            text=transcript,
            width=max_width - TEXT_PAD_X * 2,
            state=tk.NORMAL if transcript else tk.HIDDEN,
        )
        root.update_idletasks()
        full_width, full_height = transcript_size(
            canvas.bbox(text) if transcript else None, screen_width
        )
        expansion = eased(drop)
        view_width = round(WIDTH + (full_width - WIDTH) * expansion)
        view_height = round(HEIGHT + (full_height - HEIGHT) * expansion)
        canvas.config(width=view_width, height=view_height)
        root.geometry(bottom_center(root, view_width, view_height))
        canvas.itemconfigure(text, width=max(1, view_width - TEXT_PAD_X * 2))
        canvas.coords(text, view_width / 2, TEXT_TOP)

        smoothed = smoothed * 0.6 + audio_level(audio_path, sensitivity) * 0.4
        phase += 0.55
        for index, bar in enumerate(bars):
            movement = 0.75 + 0.25 * math.sin(phase + index * 0.9)
            height = 4 + smoothed * 16 * movement
            progress = cascade_progress(drop, index)
            x = view_width / 2 + (index - 2) * BAR_SPACING
            y = HEIGHT / 2 + (view_height - BAR_BOTTOM_PAD - HEIGHT / 2) * progress
            canvas.coords(bar, x, y - height / 2, x, y + height / 2)
        root.after(50, animate)

    animate()
    root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) in {3, 4}:
        main(
            Path(sys.argv[1]),
            float(sys.argv[2]),
            Path(sys.argv[3]) if len(sys.argv) == 4 else None,
        )
    else:
        raise SystemExit("usage: visualizer.py AUDIO_FILE SENSITIVITY [QUESTION_FILE]")
