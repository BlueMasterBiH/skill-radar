"""dental-gold, dynamic cut: the same storyboard plus camera cuts, depth words, drawn b-roll and sound effects.

usage: dynamic.py <workdir> [--preview 2,5,9] [--out file]

Needs in <workdir>: clip.mp4, words.json, face.json, storyboard.json, and the person matte in .arc/matte/NNNN.png
(made by arcads-cut's prep.py: `<arcads PY> <arcads>/engine/prep.py <workdir>/clip.mp4 <workdir>/.arc`).
Writes dental-gold.mp4 (with the voice) and sfx.json (sound-effect cues that music.py mixes in).

Extra storyboard events on top of the dental-gold ones:
  {"type": "depth", "at", "to", "text"}                 giant gold word slammed in BEHIND the speaker
  {"type": "broll", "at", "to", "kind", "marks": {...}, "title"}   full-screen drawn b-roll, speaker in a window
       kinds and marks are listed in broll.py
  {"type": "cam", "at", "shot": "W"|"M"|"C", "mode": "cut"|"push"}  force a camera change (else automatic)
Camera: wide (W), medium (M) and close (C) framings cut on sentence ends, about every 2-3 s, with a fast
zoom-blur push now and then, like a two-camera shoot. Depth words force a wide or medium framing.
"""
import argparse
import json
import math
import re
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parent))
import elements as E  # noqa: E402
import render as R  # noqa: E402
import broll as B  # noqa: E402

W, H, FPS = 1080, 1920, 30
SHOTS = {"W": (1.0, None), "M": (1.2, 560), "C": (1.48, 640)}
CYCLE = ["M", "C", "W", "M", "C", "W", "C", "M", "W"]


def smoothstep(x):
    x = E.clamp(x)
    return x * x * (3 - 2 * x)


def resolver():
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
    return T


def zoom_blur(img, cx, cy, amount, n=7):
    if abs(amount) < 0.004:
        return img
    acc = np.zeros_like(img)
    for k in range(n):
        s = 1 + amount * (k / (n - 1) - 0.5)
        M = np.float32([[s, 0, cx - s * cx], [0, s, cy - s * cy]])
        acc += cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return acc / n


# ---------- depth words ----------
class Depth:
    """the speaker slides right over the clean wall plate; the word sits left, partly behind her shoulder"""
    SHIFT = 250

    def __init__(self, t0, t1, text, cy=420, left=56, maxw=560):
        self.t0, self.t1, self.left = t0, t1, left
        parts = text.split()
        best = None
        for size in range(380, 90, -10):                 # largest size that fits in at most two lines
            f = B.anton(size)
            for cut in range(1, len(parts) + 1):
                lines = [" ".join(parts[:cut])] + ([" ".join(parts[cut:])] if cut < len(parts) else [])
                if all(E.text_w(l, f) <= maxw for l in lines):
                    best = (size, f, lines)
                    break
            if best:
                break
        size, f, lines = best or (100, B.anton(100), [text])
        lh = int(size * 0.98)
        w = int(max(E.text_w(l, f) for l in lines)) + 60
        h = lh * len(lines) + int(size * 0.4) + 40
        self.cy = max(cy, 150 + h / 2)
        mask = Image.new("L", (w, h), 0)
        md = ImageDraw.Draw(mask)
        for k, l in enumerate(lines):
            md.text((30, 10 + k * lh), l, font=f, fill=255)
        y = np.linspace(0, 1, h)[:, None, None]
        grad = (np.array(E.GOLD_HI, np.float32) * (1 - y) + np.array([176, 136, 62], np.float32) * y) * np.ones((1, w, 1))
        im = Image.fromarray(np.dstack([grad, np.asarray(mask)]).astype(np.uint8), "RGBA")
        sh = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        sh.putalpha(mask.filter(ImageFilter.GaussianBlur(16)).point(lambda v: int(v * 0.55)))
        self.im = Image.alpha_composite(sh, im)

    def active(self, t):
        return self.t0 - 0.02 <= t <= self.t1 + 0.4

    def shift(self, t):
        if not self.active(t):
            return 0.0
        return self.SHIFT * E.spring(t, self.t0 - 0.05, 0.4) * (1 - E.ease_out(t, self.t1, 0.35))

    def draw(self, img, t):
        if not self.active(t):
            return
        s = E.spring(t, self.t0, 0.32)
        ex = E.ease_out(t, self.t1, 0.22)
        hold = 0.04 * E.clamp((t - self.t0) / max(0.1, self.t1 - self.t0))
        sc = (1.7 - 0.7 * s + hold) * (1 + 0.1 * ex)
        a = E.clamp((t - self.t0) / 0.08) * (1 - ex)
        cx = self.left + self.im.width / 2
        if s < 0.85:                                  # motion trail while it slams in
            for k, g in ((1, 0.25), (2, 0.12)):
                E.put(img, self.im, cx, self.cy, sc * (1 + 0.12 * k * (1 - s)), a * g)
        E.put(img, self.im, cx, self.cy, sc, a)


# ---------- kinetic captions ----------
_wc = {}


def word_im(text, gold):
    k = (text, gold)
    if k not in _wc:
        f = E.font(66, "ExtraBold")
        w = int(E.text_w(text, f)) + 30
        im = Image.new("RGBA", (w, 110), (0, 0, 0, 0))
        ImageDraw.Draw(im).text((15, 18), text, font=f, fill=(E.GOLD_HI if gold else E.WHITE) + (255,))
        sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
        sh.putalpha(im.split()[3].filter(ImageFilter.GaussianBlur(7)).point(lambda v: int(v * 0.85)))
        _wc[k] = Image.alpha_composite(sh, im)
    return _wc[k]


def draw_kinetic(img, t, groups, y):
    for k, g in enumerate(groups):
        end = groups[k + 1][0]["s"] - 0.02 if k + 1 < len(groups) else g[-1]["e"] + 0.6
        end = min(end, g[-1]["e"] + 0.7)
        if not (g[0]["s"] - 0.06 <= t < end):
            continue
        txts = [w["w"].strip(",.?!") for w in g]
        ims = [word_im(x, False) for x in txts]
        gap = -8
        total = sum(i.width for i in ims) + gap * (len(ims) - 1)
        x = 540 - total / 2
        cur = max((i for i, w in enumerate(g) if w["s"] - 0.05 <= t), default=0)
        for i, (w, im) in enumerate(zip(g, ims)):
            if w["s"] - 0.05 <= t:
                gold = i == cur and t < w["e"] + 0.25
                pi = word_im(txts[i], gold)
                s = E.spring(t, w["s"] - 0.05, 0.28)
                sc = (0.7 + 0.3 * s) * (1.06 if gold else 1.0)
                E.put(img, pi, x + im.width / 2, y + (1 - s) * 18, sc, E.clamp((t - w["s"] + 0.05) / 0.08))
            x += im.width + gap
        return


# ---------- two-line sentence captions ----------
# a caption never ends on these (prepositions, conjunctions, clitics, determiners) ...
BAD_END = set("""kod na u sa s od za bez do iz o po pri prema kroz uz oko i a ali da ili nego jer ako kada kad
koji koja koje kojem što sto ti se mi li su ću će ćeš ne ni tvoj tvog tvoja tvoje svoj svog moj ova ove ovaj ovo
taj ta te koliko kakvom kakav kako zato pa već još tek okviru zajedno umjesto prvi prva prvu prvo drugi druga drugu treći""".split())
# a new line or caption reads best when it starts with one of these
GOOD_START = set("i a da ali jer ako nego ili kada kad koji koje u na sa bez od za zato pa ne".split())
ADJ_END = ("nog", "nim", "nih", "noj", "nom", "vne", "tne", "kog", "kim", "kih", "ske", "ski", "skog", "ivne")


def _weak_end(w, nxt=None):
    k = _k(w)
    if k in BAD_END:
        return True
    return nxt is not None and not re.search(r"[,.?!;:]$", w) and k.endswith(ADJ_END)
# ... and never starts on these (clitics that lean on the word before)
BAD_START = set("je se ti mi li su ću će ćeš ga joj ih nam vam".split())
LINE_MAX = 28


def _k(w):
    return re.sub(r"[^\wčćšđž]", "", w.lower())


def _line_split(ws, with_cost=False):
    """best split of a caption into two lines: balanced, no line ending on a weak word"""
    txt = [w["w"] for w in ws]
    total = len(" ".join(txt))
    if total <= LINE_MAX - 6 or len(ws) == 1:
        return [" ".join(txt)] if not with_cost else ([" ".join(txt)], 0)
    best, bk = None, 1
    for k in range(1, len(ws)):
        a, b = " ".join(txt[:k]), " ".join(txt[k:])
        c = abs(len(a) - len(b)) + (40 if max(len(a), len(b)) > LINE_MAX + 2 else 0)
        c += 20 if _weak_end(txt[k - 1], txt[k]) else 0
        c += 20 if _k(txt[k]) in BAD_START else 0
        c -= 5 if _k(txt[k]) in GOOD_START else 0
        c -= 6 if re.search(r"[,;:]$", txt[k - 1]) else 0
        if best is None or c < best:
            best, bk = c, k
    lines = [" ".join(txt[:bk]), " ".join(txt[bk:])]
    return (lines, best) if with_cost else lines


def captions_from_file(path, words):
    """captions.txt: one caption per line, ' / ' marks the line break; words must follow words.json in order"""
    caps, i = [], 0
    for raw in open(path, encoding="utf-8").read().splitlines():
        if not raw.strip():
            continue
        lines = [l.strip() for l in raw.split(" / ")]
        toks = " ".join(lines).split()
        ws = words[i:i + len(toks)]
        if len(ws) < len(toks) or [_k(w["w"]) for w in ws] != [_k(x) for x in toks]:
            raise SystemExit(f"CAPTIONS_MISMATCH at '{raw}': expected '{' '.join(w['w'] for w in words[i:i + len(toks)])}'")
        caps.append({"s": ws[0]["s"], "e": ws[-1]["e"], "lines": lines})
        i += len(toks)
    if i != len(words):
        raise SystemExit(f"CAPTIONS_MISMATCH: captions.txt stops at word {i} of {len(words)}")
    return caps


def sentence_captions(words):
    """split the words into captions of at most two lines that end where the meaning allows"""
    n = len(words)
    INF = 1e9

    def seg_cost(j, k):                       # words j..k-1
        ws = words[j:k]
        txt = " ".join(w["w"] for w in ws)
        if any(re.search(r"[.?!]$", w["w"]) for w in ws[:-1]):
            return INF                        # never run across a sentence end
        if len(txt) > 2 * LINE_MAX and k - j > 1:
            return INF
        lines, lc = _line_split(ws, True)
        if len(lines) == 2 and max(len(x) for x in lines) > LINE_MAX + 2:
            return INF
        c = 0.01 * (40 - len(txt)) ** 2 + (0.6 * lc if lc > 15 else 0)
        last, first = _k(ws[-1]["w"]), _k(ws[0]["w"])
        if re.search(r"[.?!]$", ws[-1]["w"]):
            c -= 6
        elif re.search(r"[,;:]$", ws[-1]["w"]):
            c -= 3
        elif k < n and words[k]["s"] - ws[-1]["e"] > 0.3:
            c -= 2
        if k < n and _weak_end(ws[-1]["w"], words[k]["w"]):
            c += 14
        if k < n and _k(words[k]["w"]) in GOOD_START:
            c -= 2
        if first in BAD_START:
            c += 14
        if ws[-1]["e"] - ws[0]["s"] > 4.0:
            c += 4
        return c

    best = [0.0] + [INF] * n
    prev = [0] * (n + 1)
    for k in range(1, n + 1):
        for j in range(max(0, k - 12), k):
            c = best[j] + seg_cost(j, k)
            if c < best[k]:
                best[k], prev[k] = c, j
    cuts, k = [], n
    while k > 0:
        cuts.append((prev[k], k))
        k = prev[k]
    caps = []
    for j, k in reversed(cuts):
        ws = words[j:k]
        caps.append({"s": ws[0]["s"], "e": ws[-1]["e"], "lines": _line_split(ws)})
    return caps


_lc = {}


def caption_im(lines):
    key = tuple(lines)
    if key not in _lc:
        f = E.font(50, "Bold")
        lh = 66
        w = int(max(E.text_w(l, f) for l in lines)) + 60
        h = lh * len(lines) + 40
        im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        for i, l in enumerate(lines):
            d.text(((w - E.text_w(l, f)) / 2, 14 + i * lh), l, font=f, fill=E.WHITE + (255,),
                   stroke_width=3, stroke_fill=(0, 0, 0, 255))
        sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
        sh.putalpha(im.split()[3].filter(ImageFilter.GaussianBlur(6)).point(lambda v: int(v * 0.9)))
        _lc[key] = Image.alpha_composite(Image.alpha_composite(sh, sh), im)
    return _lc[key]


def draw_two_line(img, t, caps, y):
    for k, c in enumerate(caps):
        end = caps[k + 1]["s"] - 0.03 if k + 1 < len(caps) else c["e"] + 0.6
        end = min(end, c["e"] + 0.8)
        if c["s"] - 0.08 <= t < end:
            a = E.clamp((t - c["s"] + 0.08) / 0.08) * E.clamp((end - t) / 0.08)
            E.put(img, caption_im(c["lines"]), 540, y, 1.0, a)
            return


# ---------- picture in picture ----------
def pip_patch(frame, face, w=320, h=420):
    fx, fy = face[0], face[1] + 120
    cw, ch = 640, 840
    x0 = int(E.clamp(fx - cw / 2, 0, W - cw))
    y0 = int(E.clamp(fy - ch / 2, 0, H - ch))
    crop = cv2.resize(np.clip(frame[y0:y0 + ch, x0:x0 + cw], 0, 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_AREA)
    im = Image.fromarray(crop).convert("RGBA")
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, w - 1, h - 1), 34, fill=255)
    im.putalpha(mask)
    ImageDraw.Draw(im).rounded_rectangle((1, 1, w - 2, h - 2), 34, outline=E.AQUA + (255,), width=5)
    return E.with_shadow(im, 20, 0.55, 10)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workdir")
    ap.add_argument("--preview")
    ap.add_argument("--out")
    a = ap.parse_args()
    wd = Path(a.workdir)
    sb = json.load(open(wd / "storyboard.json"))
    words = json.load(open(wd / "words.json"))
    face0 = json.load(open(wd / "face.json"))
    E.FACE = tuple(face0)
    brand = sb.get("brand", {})
    if "gold" in brand:
        E.AQUA = E.RED = tuple(brand["gold"])
    if "gold_hi" in brand:
        E.GOLD_HI = tuple(brand["gold_hi"])
    base = [e for e in sb["events"] if e["type"] not in ("depth", "broll", "cam")]
    ev, lock, _ = R.build({"events": base}, words)
    T = resolver()
    depths = [Depth(T(e["at"]), T(e["to"]), e["text"]) for e in sb["events"] if e["type"] == "depth"]
    brolls = []
    for e in sb["events"]:
        if e["type"] == "broll":
            marks = {k: T(v) for k, v in e.get("marks", {}).items()}
            kw = {"title": e["title"]} if e.get("title") else {}
            brolls.append(B.KINDS[e["kind"]](T(e["at"]), T(e["to"]), marks, **kw))
    groups = R.caption_groups(words)
    two_line = sb.get("captions") == "two_line"
    caps = None
    if two_line:
        ct = wd / "captions.txt"
        if ct.exists():
            caps = captions_from_file(ct, words)
        else:
            caps = sentence_captions(words)
            ct.write_text("\n".join(" / ".join(c["lines"]) for c in caps) + "\n", encoding="utf-8")
            print("WROTE", ct, "(edit it to fix any break, then render again)", flush=True)
    subtle = sb.get("camera") == "subtle"

    # face track per frame (arcads prep), smoothed hard for a calm crop
    ft = wd / ".arc" / "face.json"
    faces = np.array(json.load(open(ft)) if ft.exists() else [face0], float)
    k = 12
    pad = np.pad(faces, ((k, k), (0, 0)), mode="edge")
    faces = np.stack([pad[i:i + 2 * k + 1].mean(0) for i in range(len(faces))])
    mdir = wd / ".arc" / "matte"
    has_matte = mdir.exists() and any(mdir.iterdir())
    pp_ = wd / ".arc" / "plate.png"
    has_plate = pp_.exists()
    plate = cv2.resize(cv2.cvtColor(cv2.imread(str(pp_)), cv2.COLOR_BGR2RGB), (W, H)) if has_plate else None
    if depths and not has_matte:
        print("WARN: no matte in .arc/matte, depth words will be drawn in front of the speaker", flush=True)

    # camera plan
    t_end = words[-1]["e"] + 0.6
    cuts, last = [], 0.0
    for i in range(1, len(words)):
        if re.search(r"[.?!]$" if subtle else r"[,.?!]$", words[i - 1]["w"]) and words[i]["s"] - last >= (3.0 if subtle else 1.9):
            cuts.append(words[i]["s"] - 0.04)
            last = words[i]["s"]
    keys = [(0.0, "W", "cut")] if subtle else [(0.0, "M", "open")]
    for j, tc in enumerate(cuts):
        if subtle:
            keys.append((tc, "M" if j % 2 == 0 else "W", "cut"))
        else:
            keys.append((tc, CYCLE[j % len(CYCLE)], "push" if j % 3 == 1 else "cut"))
    for e in sb["events"]:
        if e["type"] == "cam":
            tc = T(e["at"])
            keys = [kk for kk in keys if abs(kk[0] - tc) > 0.5] + [(tc, e["shot"], e.get("mode", "cut"))]
    for j, d in enumerate(depths):                # wide while a depth word shows, then cut back in
        keys = [kk for kk in keys if not (d.t0 - 0.6 <= kk[0] <= d.t1 + 0.6)]
        keys += [(d.t0 - 0.05, "W", "cut"), (d.t1 + 0.42, "C" if j % 2 else "M", "cut")]
    keys.sort()
    fixed = []
    for tc, shot, mode in keys:                      # depth words need room around the speaker
        if shot != "W" and any(d.t0 - 0.3 <= tc <= d.t1 or (tc <= d.t0 and d.t0 - tc < 2.5) for d in depths):
            shot = "W"
        fixed.append((tc, shot, mode))
    keys = fixed

    def cam(t):
        idx = max(i for i, kk in enumerate(keys) if kk[0] <= t + 1e-6)
        tc, shot, mode = keys[idx]
        z, ty = SHOTS[shot]
        if subtle and shot == "M":
            z, ty = 1.14, 540
        if mode == "open":
            u = B.E.ease_out(t, 0, 0.45)
            return 1.35 + (z - 1.35) * u, ty, u < 0.95
        if mode == "push" and idx:
            pz, pty = SHOTS[keys[idx - 1][1]]
            u = E.ease_out(t, tc, 0.28)
            zz = pz + (z - pz) * u
            yy = (pty or 960) + ((ty or 960) - (pty or 960)) * u
            return zz, yy, u < 0.95
        return z, ty, False

    def punch(t):
        p = 1.0
        for l in lock:
            if l.t0 - 0.1 <= t <= l.t1 + 0.4:
                p += 0.05 * E.spring(t, l.t0, 0.35) * (1 - E.ease_out(t, l.t1, 0.4))
        return p

    # sound-effect cues for music.py
    sfx = []
    for b in brolls:
        sfx += [[round(b.t0 - 0.18, 3), "whoosh", 1.0], [round(b.t1 - 0.18, 3), "whoosh", 0.6]]
    for d in depths:
        sfx.append([round(d.t0, 3), "impact", 1.0])
    for l in lock:
        sfx.append([round(l.t0, 3), "pop", 0.7])
    for e in ev:
        if type(e).__name__ in ("Card", "Checklist", "Compare", "RootCanal", "VeneerFront", "VeneerSide", "CommentBar"):
            sfx.append([round(e.t0, 3), "pop", 0.5])
    for tc, shot, mode in keys:
        if mode == "push":
            sfx.append([round(tc - 0.05, 3), "whoosh", 0.35])
    sfx.sort()

    times = [float(x) for x in a.preview.split(",")] if a.preview else None
    out_path = Path(a.out) if a.out else wd / ("preview_dyn.jpg" if times else "dental-gold.mp4")
    if sb.get("sfx") is False:
        sfx = []
    if not times:
        json.dump(sfx, open(wd / "sfx.json", "w"))
        enc = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                                "-r", str(FPS), "-i", "-", "-i", str(wd / "clip.mp4"), "-map", "0:v", "-map", "1:a?",
                                "-c:v", "libx264", "-crf", "18", "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a",
                                "aac", "-b:a", "192k", "-shortest", str(out_path)], stdin=subprocess.PIPE)
    shots = []
    prev_z = None
    for i, raw in enumerate(R.reader(wd / "clip.mp4", W, H)):
        t = i / FPS
        if times and not any(abs(t - x) < 0.5 / FPS for x in times):
            if t > max(times):
                break
            prev_z = None
            continue
        fc = faces[min(i, len(faces) - 1)]
        z, ty, moving = cam(t)
        z *= punch(t)
        fx, fy = fc[0], fc[1]
        ty_ = ty if ty is not None else fy
        x0 = E.clamp(fx - 540 / z, 0, W - W / z)
        y0 = E.clamp(fy - ty_ / z, 0, H - H / z)
        M = np.float32([[z, 0, -x0 * z], [0, z, -y0 * z]])
        frame = cv2.warpAffine(raw, M, (W, H), flags=cv2.INTER_LINEAR).astype(np.float32)
        live = [d for d in depths if d.active(t)]
        if live:
            sh = max(d.shift(t) for d in live) if has_plate else 0.0
            mt = None
            if has_matte:
                mt = cv2.imread(str(mdir / f"{min(i, len(faces) - 1):04d}.png"), cv2.IMREAD_GRAYSCALE)
            if has_plate and mt is not None:
                k_ = E.clamp(sh / 40)
                back = cv2.warpAffine(plate, M, (W, H), flags=cv2.INTER_LINEAR) * k_ + frame * (1 - k_)
            else:
                back = frame.copy()
            for d in live:
                d.draw(back, t)
            if mt is not None:
                Mp = M.copy()
                Mp[0, 2] += sh
                person = cv2.warpAffine(raw, Mp, (W, H), flags=cv2.INTER_LINEAR).astype(np.float32)
                mp_ = cv2.warpAffine(mt, Mp, (W, H), flags=cv2.INTER_LINEAR).astype(np.float32)[..., None] / 255
                frame = person * mp_ + back * (1 - mp_)
                fx += sh / z
            else:
                frame = back
        if prev_z is not None and abs(z - prev_z) / z > 0.004 and moving:
            frame = zoom_blur(frame, 540, (fy - y0) * z, (z - prev_z) / z * 3.0)
        prev_z = z
        face_scr = ((fx - x0) * z, (fy - y0) * z)
        # b-roll mix
        m, bimg, bb = 0.0, None, None
        for b in brolls:
            mm = smoothstep((t - (b.t0 - 0.09)) / 0.18) * (1 - smoothstep((t - (b.t1 - 0.09)) / 0.18))
            if mm > 0:
                m, bb = mm, b
        img = frame
        if m > 0:
            bimg = bb.draw(min(max(t, bb.t0), bb.t1))
            whip = 1 - m
            zin = 1 + 0.18 * m
            Mz = np.float32([[zin, 0, 540 - zin * 540], [0, zin, 960 - zin * 960]])
            fa = zoom_blur(cv2.warpAffine(frame, Mz, (W, H), borderMode=cv2.BORDER_REFLECT), 540, 960, 0.08 * math.sin(math.pi * m))
            zb = 1 + 0.15 * whip
            Mb = np.float32([[zb, 0, 540 - zb * 540], [0, zb, 960 - zb * 960]])
            fb = cv2.warpAffine(bimg, Mb, (W, H), borderMode=cv2.BORDER_REFLECT)
            if 0 < m < 1:
                fb = zoom_blur(fb, 540, 960, 0.08 * math.sin(math.pi * m))
            img = fa * (1 - m) + fb * m
            img += max(0.0, 1 - abs(m - 0.5) * 4) * np.array([70, 58, 34], np.float32)
            if m > 0.6:
                pa = E.clamp((m - 0.6) / 0.4)
                pp = pip_patch(frame, (face_scr[0], face_scr[1]))
                E.put(img, pp, 862, 330, 0.85 + 0.15 * E.spring(t, bb.t0, 0.35), pa)
        if m < 0.5:
            for e in ev:
                e.draw(img, t)
            for l in lock:
                l.draw(img, t)
            if two_line:
                draw_two_line(img, t, caps, 1180)
            elif not any(l.active(t) for l in lock):
                draw_kinetic(img, t, groups, E.CAP_Y)
        elif two_line:
            draw_two_line(img, t, caps, 1690)
        else:
            draw_kinetic(img, t, groups, 1690)
        out = np.clip(img, 0, 255).astype(np.uint8)
        if times:
            shots.append(Image.fromarray(out).resize((360, 640)))
            continue
        enc.stdin.write(out.tobytes())
        if i % 300 == 0:
            print("frame", i, flush=True)
    if times:
        strip = Image.new("RGB", (360 * len(shots), 640))
        for k2, s in enumerate(shots):
            strip.paste(s, (360 * k2, 0))
        strip.save(out_path, quality=88)
    else:
        enc.stdin.close()
        enc.wait()
    print("OUTPUT", out_path)


if __name__ == "__main__":
    main()
