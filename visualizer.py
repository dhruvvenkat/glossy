#!/usr/bin/env python3

import math
import sys
import tkinter as tk
from array import array
from pathlib import Path

WIDTH = 112
HEIGHT = 44
BACKGROUND = "#111827"
BAR_COLOR = "#60a5fa"


def audio_level(path):
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
    return min(1.0, rms / 6000)


def main(audio_path):
    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.attributes("-alpha", 0.92)
    x = (root.winfo_screenwidth() - WIDTH) // 2
    y = root.winfo_screenheight() - HEIGHT - 48
    root.geometry(f"{WIDTH}x{HEIGHT}+{x}+{y}")

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
        smoothed = smoothed * 0.6 + audio_level(audio_path) * 0.4
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
    if len(sys.argv) != 2:
        raise SystemExit("usage: visualizer.py AUDIO_FILE")
    main(Path(sys.argv[1]))
