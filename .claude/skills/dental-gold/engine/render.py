"""Render a dental-gold edit from a storyboard.

usage:
  render.py <workdir> [--storyboard FILE] [--out FILE] [--preview 4.2,12,30]

<workdir> is a folder made by prepare.py: clip.mp4, words.json, face.json (and intro.mp4 once rendered).
Times in the storyboard are numbers (seconds) or phrases from the script:
  "dvije vrste"        start of that phrase
  "dvije vrste+0.6"    0.6 s after it starts ("-0.1" works too)
  "dvije vrste#2"      the second time the phrase is said
  "end"                end of the clip
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import cv2
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import elements as E  # noqa: E402

HERE = Path(__file__).resolve().parent


def build(sb, words):
    E.words[:] = words
    E.T_END = words[-1]["e"] + 0.4

    def T(v):
        if isinstance(v, (int, float)):
            return float(v)
        v = v.strip()
        if v == "end":
            return E.T_END + 1
        m = re.match(r"^(.*?)(#(\d+))?([+-]\d+(\.\d+)?)?$", v)
        phrase, occ, off = m.group(1).strip(), int(m.group(3) or 1), float(m.group(4) or 0)
        t, after = None, 0.0
        for _ in range(occ):
            t = E.at(phrase, after)
            after = t + 0.01
        return t + off

    ev, lock = [], []
    for e in sb["events"]:
        k = e["type"]
        if k == "lockup":
            lock.append(E.Lockup(T(e["at"]), T(e["to"]), e["big"], e.get("above", ""), e.get("below", "")))
        elif k == "badge":
            ev.append(E.Badge(T(e["at"]), T(e["to"]), e.get("text", sb.get("brand", {}).get("badge", ""))))
        elif k == "strip":
            ev.append(E.Strip(T(e["at"]), T(e["to"]), e["text"]))
        elif k == "card":
            ev.append(E.Card(T(e["at"]), T(e["to"]), e["title"], e.get("sub", ""), icon=e.get("icon")))
        elif k == "veneer_front":
            ev.append(E.VeneerFront(T(e["at"]), T(e["to"]), e["title"]))
        elif k == "veneer_side":
            ev.append(E.VeneerSide(T(e["at"]), T(e["to"]), e["title"]))
        elif k == "root_canal":
            opt = lambda f: T(e[f]) if e.get(f) is not None else None
            ev.append(E.RootCanal(T(e["at"]), T(e["to"]), e["title"], opt("remove"), opt("clean"), opt("fill"),
                                  opt("keep")))
        elif k == "compare":
            cols = []
            for side in ("left", "right"):
                c = e[side]
                cols.append((c["title"], T(c["at"]), [(b[0].replace("\\n", "\n"), T(b[1])) for b in c.get("bullets", [])],
                             (T(c["active"][0]), T(c["active"][1])) if c.get("active") else (-1, -1)))
            ev.append(E.Compare(T(e["at"]), T(e["to"]), cols[0], cols[1]))
        elif k == "checklist":
            ev.append(E.Checklist(T(e["at"]), T(e["to"]), e["title"], [(i[0], T(i[1])) for i in e["items"]]))
        elif k == "comment":
            ev.append(E.CommentBar(T(e["at"]), T(e.get("to", "end")), e["text"]))
        else:
            sys.exit(f"unknown event type: {k}")
    intro = sb.get("intro") or {}
    intro_until = T(intro["until"]) if intro.get("until") is not None else 0.0
    return ev, lock, intro_until


def caption_groups(words):
    groups, cur = [], []
    for k, w in enumerate(words):
        gap = w["s"] - words[k - 1]["e"] if k else 0
        txt = " ".join(x["w"] for x in cur + [w])
        if cur and (len(cur) >= 3 or len(txt) > 18 or gap > 0.3 or re.search(r"[,.?!]$", cur[-1]["w"])):
            groups.append(cur)
            cur = []
        cur.append(w)
    groups.append(cur)
    return groups


def draw_caption(img, t, groups, lock):
    if any(l.active(t) for l in lock):
        return
    for k, g in enumerate(groups):
        end = groups[k + 1][0]["s"] - 0.02 if k + 1 < len(groups) else g[-1]["e"] + 0.6
        end = min(end, g[-1]["e"] + 0.7)
        if g[0]["s"] - 0.06 <= t < end:
            txt = " ".join(w["w"].strip(",.") for w in g)
            s = E.ease_out(t, g[0]["s"] - 0.06, 0.18)
            E.put(img, E_cap(txt), 540, E.CAP_Y + (1 - s) * 12, 0.94 + 0.06 * s, E.clamp((t - g[0]["s"] + 0.06) / 0.1))
            return


_cap = {}


def E_cap(text):
    from PIL import ImageDraw, ImageFilter
    if text not in _cap:
        f = E.font(56, "Bold")
        w = int(E.text_w(text, f)) + 60
        im = Image.new("RGBA", (w, 110), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((30, 20), text, font=f, fill=E.WHITE)
        a = im.split()[3].filter(ImageFilter.GaussianBlur(7)).point(lambda v: int(v * 0.85))
        sh = Image.new("RGBA", im.size, (0, 0, 0, 255))
        sh.putalpha(a)
        _cap[text] = Image.alpha_composite(sh, im)
    return _cap[text]


def reader(path, w, h):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         stdout=subprocess.PIPE, bufsize=w * h * 3)
    while True:
        b = p.stdout.read(w * h * 3)
        if len(b) < w * h * 3:
            return
        yield np.frombuffer(b, np.uint8).reshape(h, w, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir")
    ap.add_argument("--storyboard")
    ap.add_argument("--out")
    ap.add_argument("--preview", help="comma-separated times: writes a contact strip instead of a video")
    a = ap.parse_args()
    wd = Path(a.workdir)
    sb = json.load(open(a.storyboard or wd / "storyboard.json"))
    words = json.load(open(wd / "words.json"))
    face = json.load(open(wd / "face.json"))
    E.FACE = tuple(face)
    brand = sb.get("brand", {})
    if "gold" in brand:
        E.AQUA = E.RED = tuple(brand["gold"])
    if "gold_hi" in brand:
        E.GOLD_HI = tuple(brand["gold_hi"])
    ev, lock, intro_until = build(sb, words)
    groups = caption_groups(words)
    W, H, FPS = E.W, E.H, E.FPS
    clip = wd / "clip.mp4"
    intro = wd / "intro.mp4"
    use_intro = intro_until > 0 and intro.exists()
    if intro_until > 0 and not intro.exists():
        print("NOTE: no intro.mp4 in the workdir, rendering without the explainer intro", flush=True)
    XF = 0.35

    def zoom(t):
        z = 1.0 + 0.012 * np.sin(t * 0.35)
        for l in lock:
            if l.t0 - 0.1 <= t <= l.t1 + 0.4:
                z += 0.06 * E.spring(t, l.t0, 0.35) * (1 - E.ease_out(t, l.t1, 0.4))
        return z

    times = [float(x) for x in a.preview.split(",")] if a.preview else None
    out_path = Path(a.out) if a.out else wd / ("preview.jpg" if times else "dental-gold.mp4")
    enc = None
    if not times:
        enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                                "-r", str(FPS), "-i", "-", "-i", str(clip), "-map", "0:v", "-map", "1:a?", "-c:v",
                                "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac",
                                "-b:a", "192k", "-shortest", str(out_path)], stdin=subprocess.PIPE)
    pan = reader(intro, W, H) if use_intro else None
    shots = []
    for i, raw in enumerate(reader(clip, W, H)):
        t = i / FPS
        pf = next(pan, None) if pan else None
        if times and not any(abs(t - x) < 0.5 / FPS for x in times):
            if t > max(times):
                break
            continue
        z = zoom(t)
        cw, ch = int(W / z), int(H / z)
        cx, cy = E.FACE[0], E.FACE[1] + 250
        x0 = int(E.clamp(cx - cw / 2, 0, W - cw))
        y0 = int(E.clamp(cy - ch / 2, 0, H - ch))
        img = cv2.resize(raw[y0:y0 + ch, x0:x0 + cw], (W, H), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        m = 0.0
        if use_intro:
            m = 1.0 if t <= intro_until else E.clamp(1 - (t - intro_until) / XF)
            if m > 0 and pf is not None:
                img = img * (1 - m) + pf.astype(np.float32) * m
        if m < 1:
            draw_caption(img, t, groups, lock)
            for e in ev:
                e.draw(img, t)
            for l in lock:
                l.draw(img, t)
        frame = np.clip(img, 0, 255).astype(np.uint8)
        if times:
            shots.append(Image.fromarray(frame).resize((360, 640)))
            continue
        enc.stdin.write(frame.tobytes())
        if i % 300 == 0:
            print("frame", i, flush=True)
    if times:
        strip = Image.new("RGB", (360 * len(shots), 640))
        for k, s in enumerate(shots):
            strip.paste(s, (360 * k, 0))
        strip.save(out_path, quality=88)
    else:
        enc.stdin.close()
        enc.wait()
    print("OUTPUT", out_path)


if __name__ == "__main__":
    main()
