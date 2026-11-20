"""/liquid-glass: everything is made of liquid glass that really refracts the footage. A glass caption capsule
jelly-stretches to every phrase; on each keyword a glass droplet splits off it, rises through the frame and
blooms into a glass card holding the label, melting into the previous card as it takes over; on the big beat a
glass lens sweeps across the speaker and magnifies him; the payoff card catches a light sweep."""
import math

import numpy as np

import core as c
import fx
from plan import Plan, norm

GOLD = (255, 204, 128)
CAP_Y, CAP_H = 1390, 136
CARD_Y = 300


class Style:
    def __init__(self, clip, opts):
        self.clip = clip
        self.P = P = Plan(clip, opts)
        self.fr = c.Framing(clip, 1215)
        B = P.beats
        big = B[min(2, len(B) - 1)]
        self.lens_t = big["key_t"]
        self.cam = fx.Cam(self.fr, [
            {"t": 0, "zoom": 1.0},
            {"t": B[0]["key_t"], "zoom": 1.05, "ease": "spring", "freq": 1.5, "damp": 0.6},
            {"t": B[1]["t0"], "zoom": 1.0, "dy": -12, "dur": 0.7},
            {"t": self.lens_t - 0.25, "zoom": 1.12, "dy": 0, "ease": "smooth", "dur": 0.8},
            {"t": B[-1]["t0"], "zoom": 1.03, "ease": "spring", "freq": 1.4, "damp": 0.7}], drift=4)
        self.groups = []
        for b in B:
            ws = b["words"]
            for n in range(0, len(ws), 2):
                g = ws[n:n + 2]
                self.groups.append({"t": g[0]["s"] if self.groups else 0.0, "words": [w["w"].strip(" ,.") for w in g]})
        self._tx = {}
        self._ln = {}

    # ---- type ----
    def tx(self, s, key, size, color=(255, 255, 255), wght=None, tracking=0.0):
        kk = (s, key, size, color, wght, tracking)
        if kk not in self._tx:
            self._tx[kk] = fx.text(s, key, size, color, wght, tracking)
        return self._tx[kk]

    def line(self, words):
        kk = tuple(words)
        if kk not in self._ln:
            ps = [self.tx(w, "grotesk", 72, GOLD if norm(w) in self.P.hot else (255, 255, 255), 780) for w in words]
            gap = 16
            w = sum(p.shape[1] for p in ps) + gap * (len(ps) - 1)
            out = c.layer(max(p.shape[0] for p in ps), w)
            x = 0
            for p in ps:
                c.over(out, p, x, 0)
                x += p.shape[1] + gap
            self._ln[kk] = out
        return self._ln[kk]

    # ---- liquid shapes ----
    def capsule(self, t):
        gs = self.groups
        j = max([n for n, g in enumerate(gs) if g["t"] <= t] or [0])
        tg = gs[j]["t"]
        nw = self.line(gs[j]["words"]).shape[1] + 124
        if j == 0:
            e = fx.spring(t, 0.0, 2.3, 0.5)
            return j, tg, nw * e, CAP_H * e
        pw = self.line(gs[j - 1]["words"]).shape[1] + 124
        e = fx.spring(t, tg, 2.4, 0.4)
        h = CAP_H * (1 - 0.17 * math.sin(math.pi * c.clamp((t - tg) / 0.3)))
        return j, tg, pw + (nw - pw) * e, h

    def cards(self, t):
        B = self.P.beats
        out = []
        for j, b in enumerate(B):
            ts = b["key_t"] - 0.1
            if t < ts:
                continue
            last = j == len(B) - 1
            lab = self.tx(b["label"], "grotesk", 92 if last else 76, (255, 255, 255), 800)
            cw, ch = lab.shape[1] + (180 if last else 130), (224 if last else 180)
            e = fx.spring(t, ts, 1.8, 0.62)
            g = fx.spring(t, ts + 0.16, 2.1, 0.42)
            y = c.lerp(CAP_Y, CARD_Y + (20 if last else 0), e)
            w, h = c.lerp(150, cw, g), c.lerp(150, ch, g)
            q, tn = 0.0, 9.0
            if not last:
                tn = B[j + 1]["key_t"] - 0.1
                q = fx.expo(t, tn + 0.3, 0.38)
            w, h = w * (1 - q), h * (1 - q)
            if w > 6 and h > 6:
                out.append({"b": b, "x": 540, "y": y - 30 * q, "w": w, "h": h, "lab": lab, "g": g, "q": q,
                            "last": last, "ts": ts, "tn": tn})
        return out

    # ---- frame ----
    def frame(self, i, rgb, alpha):
        t = i / self.clip.fps
        img, _ = self.cam.place(i, t, rgb, alpha)
        img = c.grade(img, contrast=1.08, sat=1.04, warm=0.012)
        j, tg, cw_, chh = self.capsule(t)
        cards = self.cards(t)
        shapes = [(540, CAP_Y, cw_, chh)] + [(d["x"], d["y"], d["w"], d["h"], min(64, d["h"] / 2)) for d in cards]
        e0 = fx.expo(t, 0.05, 0.55)  # opening: a big glass sphere melts down into the caption capsule
        D = 900 * (1 - e0) + 40
        if D > 70:
            shapes.append((540, c.lerp(780, CAP_Y, e0 ** 1.4), D, D))
        shapes = [s for s in shapes if s[2] > 2 and s[3] > 2]
        if shapes:
            x0 = int(max(0, min(s[0] - s[2] / 2 for s in shapes) - 70))
            x1 = int(min(c.W, max(s[0] + s[2] / 2 for s in shapes) + 70))
            y0 = int(max(0, min(s[1] - s[3] / 2 for s in shapes) - 70))
            y1 = int(min(c.H, max(s[1] + s[3] / 2 for s in shapes) + 70))
            local = [(s[0] - x0, s[1] - y0, *s[2:]) for s in shapes]
            m = fx.liquid(x1 - x0, y1 - y0, local, k=46)
            fx.glass_shadow(img, m, x0, y0, blur=30, a=0.28, dy=20)
            c.over(img, fx.glass(img, m, x0, y0, bevel=c.lerp(300, 46, e0), refract=1.4, magnify=c.lerp(0.16, 0.04, e0),
                                 frost=1.6, chroma=0.8, tint_a=0.06), x0, y0)
        # caption text
        new = self.line(self.groups[j]["words"])
        a_in = c.clamp((t - tg) / 0.14) if j else c.clamp((t - 0.08) / 0.14)
        dy = (1 - fx.expo(t, tg, 0.3)) * 26 if j else 0
        if j:
            old = self.line(self.groups[j - 1]["words"])
            fx.put(img, old, 540, CAP_Y - 18 * c.clamp((t - tg) / 0.12) + 2, opacity=1 - c.clamp((t - tg) / 0.1),
                   shadow=0.35, sblur=10, sdy=3)
        fx.put(img, new, 540, CAP_Y + dy + 2, opacity=a_in, mv=(0, -dy * 0.5), shadow=0.35, sblur=10, sdy=3)
        # card labels
        for d in cards:
            op = c.clamp((d["g"] - 0.55) / 0.35) * (1 - c.clamp((t - d["tn"] - 0.1) / 0.12))
            if op <= 0:
                continue
            sub = d["b"].get("sub", "")
            s = 0.92 + 0.08 * min(1.0, d["g"])
            ly = d["y"] + (14 if sub else 0)
            fx.put(img, d["lab"], d["x"], ly, s, opacity=op, shadow=0.3, sblur=12, sdy=4)
            if sub:
                fx.put(img, self.tx(sub, "mono", 24, (255, 255, 255), 560, 0.22), d["x"],
                       ly - d["lab"].shape[0] / 2 - 10, opacity=op * 0.75)
        # payoff light sweep across the last card
        for d in cards:
            if d["last"]:
                p = c.prog(t, d["ts"] + 0.45, 0.7)
                if 0 < p < 1:
                    x0, y0 = int(d["x"] - d["w"] / 2), int(d["y"] - d["h"] / 2)
                    msk = c.rrect(int(d["w"]), int(d["h"]), int(min(64, d["h"] / 2)))
                    ys, xs = fx.grid(*msk.shape)
                    u = (xs * 0.91 + ys * 0.41) / (msk.shape[1] * 0.91 + msk.shape[0] * 0.41)
                    glow = np.exp(-((u - (p * 1.4 - 0.2)) / 0.1) ** 2) * msk * 0.45
                    fx.add(img, np.dstack([glow * 255] * 4), x0, y0)
        # the lens: a big glass disc sweeps across the speaker on the big beat
        p = c.prog(t, self.lens_t - 0.5, 1.45)
        if 0 < p < 1:
            u = 2 * p - 1
            e = 0.5 + 0.5 * (0.35 * u + 0.65 * u ** 3)  # fast in, lingers over the face, fast out
            D = 520
            cx, cy = c.lerp(-300, 1380, e), c.lerp(900, 600, p)
            m = c.rrect(D, D, D // 2)
            x0, y0 = int(cx - D / 2), int(cy - D / 2)
            fx.glass_shadow(img, m, x0, y0, blur=40, a=0.3, dy=26)
            c.over(img, fx.glass(img, m, x0, y0, bevel=170, refract=1.25, magnify=0.2, frost=0.0, chroma=0.55,
                                 tint_a=0.03, spec=1.0), x0, y0)
        img = c.bloom(img, 215, 26, 0.25)
        img = c.vignette(img, 0.32)
        return c.grain(img, i, 2.5)
