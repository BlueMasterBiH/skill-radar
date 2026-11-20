"""Render one arcads-cut style.

usage:
  python render.py <style> <clip.mp4> <cache_dir> <out.mp4>            full render (silent; --audio keeps the sound)
  python render.py <style> <clip.mp4> <cache_dir> <sheet.jpg> --sheet  contact sheet (every 6th frame)
  python render.py <style> <clip.mp4> <cache_dir> <dir> --still 1.2,3.4  PNG check frames
  extra: --opt key=value (repeatable) passes text/colour options to the style
"""
import argparse
import importlib
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import core  # noqa: E402


def load(style, clip, opts):
    mod = importlib.import_module("styles." + style.replace("-", "_"))
    return mod.Style(clip, opts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("style")
    ap.add_argument("clip")
    ap.add_argument("cache")
    ap.add_argument("out")
    ap.add_argument("--sheet", action="store_true")
    ap.add_argument("--every", type=int, default=6)
    ap.add_argument("--still")
    ap.add_argument("--opt", action="append", default=[])
    ap.add_argument("--audio", action="store_true", help="keep the clip's audio (default: silent)")
    a = ap.parse_args()
    opts = dict(o.split("=", 1) for o in a.opt)
    clip = core.Clip(a.clip, a.cache)
    st = load(a.style, clip, opts)
    t0 = time.time()

    if a.still:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        for s in a.still.split(","):
            i = min(int(round(float(s) * clip.fps)), clip.n - 1)
            img = st.frame(i, clip.frame_at(i), clip.matte(i))
            cv2.imwrite(str(out / f"{a.style}_{float(s):.2f}.png"), cv2.cvtColor(core.to_u8(img), cv2.COLOR_RGB2BGR))
        print("stills", out, f"{time.time() - t0:.1f}s")
        return

    if a.sheet:
        frames, labels = [], []
        for i, rgb in clip.frames():
            if i % a.every:
                continue
            frames.append(st.frame(i, rgb, clip.matte(i)))
            labels.append(f"{i / clip.fps:.2f}s")
        core.contact_sheet(frames, labels, a.out)
        print("sheet", a.out, f"{time.time() - t0:.1f}s")
        return

    w = core.Writer(a.out, clip, audio=a.audio)
    for i, rgb in clip.frames():
        w.write(st.frame(i, rgb, clip.matte(i)))
    w.close()
    print("video", a.out, f"{time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
