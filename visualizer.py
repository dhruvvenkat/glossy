#!/usr/bin/env python3

import math
import re
import subprocess
import sys
import tkinter as tk
from array import array
from pathlib import Path

WIDTH = 112
HEIGHT = 44
QUESTION_WIDTH = 560
BACKGROUND = "#111827"
BAR_COLOR = "#60a5fa"
TEXT_COLOR = "#f9fafb"


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


def render_question(root, question):
    for child in root.winfo_children():
        child.destroy()
    width = max(220, min(QUESTION_WIDTH, root.winfo_screenwidth() - 64))
    label = tk.Label(
        root,
        text=question,
        bg=BACKGROUND,
        fg=TEXT_COLOR,
        font=("Sans", 11),
        justify=tk.CENTER,
        wraplength=width - 28,
        padx=14,
        pady=10,
    )
    label.pack(fill=tk.BOTH, expand=True)
    root.update_idletasks()
    height = max(HEIGHT, label.winfo_reqheight())
    root.geometry(bottom_center(root, width, height))


def main(audio_path, sensitivity, question_path=None):
    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.attributes("-alpha", 0.92)
    root.geometry(bottom_center(root, WIDTH, HEIGHT))

    canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg=BACKGROUND, highlightthickness=0)
    canvas.pack()
    bars = [
        canvas.create_line(x, 19, x, 25, fill=BAR_COLOR, width=6, capstyle=tk.ROUND)
        for x in (32, 44, 56, 68, 80)
    ]
    smoothed = 0.0
    phase = 0.0

    def animate():
        nonlocal smoothed, phase
        if question_path is not None:
            try:
                question = question_path.read_text().strip()
            except OSError:
                question = ""
            if question:
                render_question(root, question)
                return
        smoothed = smoothed * 0.6 + audio_level(audio_path, sensitivity) * 0.4
        phase += 0.55
        for index, bar in enumerate(bars):
            movement = 0.75 + 0.25 * math.sin(phase + index * 0.9)
            height = 6 + smoothed * 28 * movement
            x = 32 + index * 12
            canvas.coords(bar, x, 22 - height / 2, x, 22 + height / 2)
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
