"""Prepare a raw recording for dental-gold.

The recording holds every hook one after another, then the lead, body and CTA once (the usual way the
clinic films). The script file lists those parts:

  **Hook 1:** ...
  **Hook 2:** ...
  **Lead:** ...
  **Body:** ...
  **CTA:** ...

usage: prepare.py <clip> <script.md> --out <dir> [--hooks 1,3] [--blocks "1.5-4.6,6.7-9.2,..."]

For each hook it writes <out>/hook<N>/ with clip.mp4 (hook + lead + body + CTA, 1080x1920, 30 fps),
words.json (script words timed to the speech) and face.json. Without Whisper the word times are spread
over the detected speech, so they are close but not exact; a real transcript can replace words.json.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

W, H, FPS = 1080, 1920, 30
PAD0, PAD1 = 0.25, 0.35


def run(cmd, **kw):
    return subprocess.run(cmd, check=True, **kw)


def parse_script(path):
    txt = Path(path).read_text(encoding="utf-8")
    parts = {}
    for m in re.finditer(r"^\s*\**\s*(Hook\s*\d+|Lead|Body|CTA)\s*:?\s*\**\s*:?\s*(.+)$", txt, re.M | re.I):
        name = re.sub(r"\s+", "", m.group(1)).lower()
        parts[name] = m.group(2).strip().strip("*").strip()
    hooks = sorted([k for k in parts if k.startswith("hook")], key=lambda k: int(k[4:]))
    missing = [k for k in ("lead", "body", "cta") if k not in parts]
    if not hooks or missing:
        sys.exit(f"SCRIPT_PARSE: found {list(parts)}; need Hook N lines plus Lead, Body and CTA")
    return parts, hooks


def normalise(src, dst):
    """1080x1920 at 30 fps; landscape is centre-cropped to 9:16."""
    vf = "fps=30,scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"
    run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", vf, "-c:v", "libx264", "-crf", "17", "-preset", "fast",
         "-c:a", "aac", "-b:a", "192k", str(dst)])


def silences(path, noise="-35dB", d=0.12):
    r = subprocess.run(["ffmpeg", "-i", str(path), "-af", f"silencedetect=noise={noise}:d={d}", "-f", "null", "-"],
                       capture_output=True, text=True).stderr
    st = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", r)]
    en = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", r)]
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(path)], capture_output=True, text=True).stdout.strip())
    if len(en) < len(st):
        en.append(dur)
    return list(zip(st, en)), dur


def speech(sil, a, b):
    iv, cur = [], a
    for s, e in sil:
        if e <= a or s >= b:
            continue
        if s > cur:
            iv.append((cur, min(s, b)))
        cur = max(cur, e)
    if cur < b:
        iv.append((cur, b))
    return [x for x in iv if x[1] - x[0] > 0.05]


def blocks_for(sil, dur, gap):
    iv = speech(sil, 0, dur)
    # drop short isolated bits (breaths, clicks, room noise) before merging
    keep = []
    for k, (s, e) in enumerate(iv):
        gp = s - iv[k - 1][1] if k else 99
        gn = iv[k + 1][0] - e if k + 1 < len(iv) else 99
        if e - s >= 0.8 or min(gp, gn) < 0.9:
            keep.append((s, e))
    iv = keep
    out = []
    for s, e in iv:
        if out and s - out[-1][1] < gap:
            out[-1] = (out[-1][0], e)
        else:
            out.append((s, e))
    return [b for b in out if b[1] - b[0] > 0.8]  # shorter bits are breaths, clicks, room noise


def align(text, iv):
    words = text.split()
    wt = [len(re.sub(r"\W", "", w)) + 1.5 + (2.0 if re.search(r"[,.?!]$", w) else 0) for w in words]
    total = sum(e - s for s, e in iv)
    scale = total / sum(wt)

    def real(t):
        for s, e in iv:
            if t <= e - s + 1e-9:
                return s + t
            t -= e - s
        return iv[-1][1]
    out, c = [], 0.0
    for w, x in zip(words, wt):
        d = x * scale
        tail = 2.0 * scale if re.search(r"[,.?!]$", w) else 0
        out.append({"w": w, "s": round(real(c + 0.01), 3), "e": round(real(c + d - tail), 3)})
        c += d
    return out


def face_box(path):
    cas = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    cap = cv2.VideoCapture(str(path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 300
    boxes = []
    for k in range(20):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(n * (k + 0.5) / 20))
        ok, fr = cap.read()
        if not ok:
            continue
        g = cv2.cvtColor(cv2.resize(fr, None, fx=0.5, fy=0.5), cv2.COLOR_BGR2GRAY)
        det = cas.detectMultiScale(g, 1.1, 6, minSize=(60, 60))
        if len(det):
            x, y, w, h = max(det, key=lambda d: d[2] * d[3]) * 2
            boxes.append((x + w / 2, y + h / 2 + 0.08 * h, w, h * 1.35))
    if not boxes:
        print("WARN: no face found, using a default position", flush=True)
        return [540, 520, 220, 300]
    return [round(float(v), 1) for v in np.median(np.array(boxes), axis=0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip")
    ap.add_argument("script")
    ap.add_argument("--out", required=True)
    ap.add_argument("--hooks", help="which hooks to build, e.g. 1,3 (default: all)")
    ap.add_argument("--blocks", help="manual speech blocks in order: hook1,hook2,...,lead,body,cta as a-b,c-d,...")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    parts, hooks = parse_script(a.script)
    order = hooks + ["lead", "body", "cta"]
    norm = out / "source_1080.mp4"
    if not norm.exists():
        print("normalising the clip...", flush=True)
        normalise(a.clip, norm)
    sil, dur = silences(norm)
    if a.blocks:
        blocks = [tuple(map(float, b.split("-"))) for b in a.blocks.split(",")]
    else:
        coarse, _ = silences(norm, d=0.35)
        blocks = None
        for gap in (1.0, 1.3, 0.8, 1.6, 0.65, 2.0, 2.5):
            bl = blocks_for(coarse, dur, gap)
            if len(bl) == len(order):
                blocks = bl
                break
        if blocks is None:
            bl = blocks_for(coarse, dur, 1.0)
            print("BLOCKS_MISMATCH: expected", len(order), "parts", order, "but found", len(bl), "speech blocks:")
            for s, e in bl:
                print(f"  {s:.2f}-{e:.2f}")
            print("Pass --blocks with one a-b range per part, in order.")
            sys.exit(2)
    if len(blocks) != len(order):
        sys.exit(f"--blocks needs {len(order)} ranges ({', '.join(order)})")
    sec = dict(zip(order, blocks))
    for k in order:
        print(f"{k:6s} {sec[k][0]:6.2f}-{sec[k][1]:6.2f}  {parts[k][:60]}", flush=True)
    aligned = {k: align(parts[k], speech(sil, *sec[k])) for k in order}
    want = [f"hook{n}" for n in a.hooks.split(",")] if a.hooks else hooks
    for h in want:
        seq = [h, "lead", "body", "cta"]
        vd = out / h
        vd.mkdir(exist_ok=True)
        words, off, filt, segs = [], 0.0, [], []
        for j, p in enumerate(seq):
            s0, s1 = max(0, sec[p][0] - PAD0), min(dur, sec[p][1] + PAD1)
            for w in aligned[p]:
                words.append({"w": w["w"], "s": round(w["s"] - s0 + off, 3), "e": round(w["e"] - s0 + off, 3)})
            filt.append(f"[0:v]trim={s0}:{s1},setpts=PTS-STARTPTS[v{j}];[0:a]atrim={s0}:{s1},asetpts=PTS-STARTPTS[a{j}];")
            segs.append(f"[v{j}][a{j}]")
            off += s1 - s0
        fc = "".join(filt) + "".join(segs) + f"concat=n={len(seq)}:v=1:a=1[v][a]"
        run(["ffmpeg", "-v", "error", "-y", "-i", str(norm), "-filter_complex", fc, "-map", "[v]", "-map", "[a]",
             "-c:v", "libx264", "-crf", "17", "-preset", "fast", "-r", "30", "-c:a", "aac", "-b:a", "192k",
             str(vd / "clip.mp4")])
        json.dump(words, open(vd / "words.json", "w"), ensure_ascii=False, indent=0)
        json.dump(face_box(vd / "clip.mp4"), open(vd / "face.json", "w"))
        (vd / "script.txt").write_text(" ".join(parts[p] for p in seq), encoding="utf-8")
        print(f"VERSION {vd}  {off:.1f}s  {len(words)} words", flush=True)
    print("PREPARE_DONE", out)


if __name__ == "__main__":
    main()
