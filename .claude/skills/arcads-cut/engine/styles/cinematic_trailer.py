"""/cinematic-trailer: the clip cut like a movie trailer. A white flash-frame opens into the shot as letterbox bars spring shut, under a rich teal-orange film grade with heavy grain and gate weave; anamorphic flares streak across on the beats; a hard cut to a black title card where thin serif caps slowly track out; back to the shot through a light leak; a rack focus melts the wall on the big word; and on the payoff the bars squeeze to a widescreen strip over an end title."""
import math

import cv2
import numpy as np

import core as c
import fx
from plan import Plan

BAR = 236                       # letterbox bar height while the shot plays
SCOPE = (c.H - c.W / 1.85) / 2  # bar height of the 1.85:1 widescreen strip on the payoff
TEAL = np.float32([-0.06, 0.02, 0.07])
ORANGE = np.float32([0.07, 0.025, -0.05])
BLUE_FL = (110, 175, 255)
AMBER_FL = (255, 168, 86)


def leak_layer(p, seed=3):
    """Organic warm light leak (RGB, 0..255-ish) drifting in from the right as p goes 0 -> 1."""
    h, w = 240, 135
    ys, xs = fx.grid(h, w)
    out = np.zeros((h, w, 3), np.float32)
    rng = np.random.default_rng(seed)
    blobs = [((1.15, 0.22), (255, 110, 40), 0.42), ((1.3, 0.55), (255, 64, 28), 0.36), ((1.05, 0.8), (255, 176, 80), 0.3),
             ((1.4, 0.38), (255, 140, 56), 0.46)]
    for (bx, by), col, r in blobs:
        cx = (bx - 0.62 * p - 0.1 * rng.random()) * w
        cy = (by + 0.06 * math.sin(p * 3 + bx)) * h
        rr = r * w * (0.85 + 0.4 * p)
        g = np.exp(-(((xs - cx) / rr) ** 2 + ((ys - cy) / (rr * 1.6)) ** 2))
        out += g[..., None] * np.float32(col)
    return cv2.resize(out, (c.W, c.H), interpolation=cv2.INTER_CUBIC)


class Style:
    def __init__(self, clip, opts):
        self.clip = clip
        self.P = P = Plan(clip, opts)
        self.fr = c.Framing(clip, 1215)
        self.N = min(clip.n, len(clip.face))
        self.dur = self.N / clip.fps
        B = P.beats
        n = len(B)
        # which beat becomes the black title card: the roomiest middle beat
        mids = list(range(1, n - 1)) or list(range(min(1, n - 1), n))
        self.tc = max(mids, key=lambda k: B[k]["t1"] - B[k]["key_t"])
        tb = B[self.tc]
        self.tc_t = tb["key_t"]
        self.tc_end = self.tc_t + c.clamp(0.55 * (tb["t1"] - tb["key_t"]), 0.55, 0.85)
        self.fin = n - 1 if n - 1 != self.tc else None
        self.fin_t = B[-1]["key_t"] if self.fin is not None else self.dur + 1
        rest = [k for k in range(n) if k not in (0, self.tc, self.fin)]
        self.dof_t = B[rest[-1]]["key_t"] if rest else self.dur + 1     # rack focus beat
        self.cuts = [self.tc_end]
        fcy = float(np.median([self.fr.face_box(i)[1] for i in range(0, self.N, 5)]))
        self.scope_dy = c.H / 2 - fcy + 4
        self.cam = fx.Cam(self.fr, self.camera_keys(), drift=1.5)
        self.grain_rng = {}

    def camera_keys(self):
        k = [{"t": 0.0, "zoom": 1.15},
             {"t": 0.0, "zoom": 1.02, "ease": "expo", "dur": 0.75},
             {"t": 0.75, "zoom": 1.08, "ease": "smooth", "dur": max(0.3, self.tc_t - 0.75)},
             {"t": self.tc_end, "zoom": 1.12, "rot": -1.1, "dx": 26, "ease": "cut"},
             {"t": self.tc_end, "zoom": 1.15, "rot": -0.4, "dx": 10, "ease": "smooth",
              "dur": max(0.3, min(self.fin_t, self.dur) - self.tc_end)}]
        if self.fin_t < self.dur:
            k += [{"t": self.fin_t - 0.05, "zoom": 1.06, "rot": 0.0, "dx": 0, "dy": self.scope_dy, "ease": "expo",
                   "dur": 0.55},
                  {"t": self.fin_t + 0.5, "zoom": 1.0, "ease": "smooth", "dur": 1.2}]
        return k

    # ---- look ----
    def grade(self, img, a):
        x = img / 255
        a3 = cv2.GaussianBlur(a, (0, 0), 3)[..., None]
        lum = x @ np.float32([0.299, 0.587, 0.114])
        # the wall: cooled, desaturated and pushed down; the speaker: warm skin
        wall = lum[..., None] + (x - lum[..., None]) * 0.35
        wall = wall * np.float32([0.72, 0.86, 0.95]) + np.float32([0.0, 0.012, 0.03])
        skin = lum[..., None] + (x - lum[..., None]) * 1.12
        x = skin * a3 + wall * (1 - a3)
        lum = x @ np.float32([0.299, 0.587, 0.114])
        x = x + ((1 - lum) ** 2)[..., None] * TEAL + (lum ** 2)[..., None] * ORANGE
        x = np.clip(x, 0, 1)
        x = x * x * (3 - 2 * x) * 0.55 + x * 0.45          # filmic S-curve
        x = x * 0.95 + np.float32([0.0, 0.018, 0.026])      # teal-lifted blacks
        return x * 255

    def grain(self, img, i, amt):
        rng = np.random.default_rng(1000 + i)
        g = rng.normal(0, 1, (c.H // 2, c.W // 2)).astype(np.float32)
        g = cv2.resize(g, (c.W, c.H), interpolation=cv2.INTER_LINEAR)
        lum = img.mean(2, keepdims=True) / 255
        return img + g[..., None] * amt * (0.55 + 0.9 * lum * (1 - lum) * 2)

    def bars(self, t):
        """Bar height: springs shut at the open, squeezes to scope on the payoff."""
        h = BAR * fx.spring(t, 0.04, 1.7, 0.62)
        if t >= self.fin_t - 0.05:
            h = c.lerp(h, SCOPE, fx.expo(t, self.fin_t - 0.05, 0.55))
        return h

    # ---- pieces ----
    def title_card(self, i, t):
        B = self.P.beats
        b = B[self.tc]
        img = np.zeros((c.H, c.W, 3), np.float32) + np.float32([4, 5, 6])
        u = c.prog(t, self.tc_t, self.tc_end - self.tc_t)
        lines = [b["label"].upper()]
        size = min(104, c.fit_size(lines[0], "display", 950, 420, 0.3))
        ws = lines[0].split()
        if size < 80 and len(ws) > 1:  # long label: stack it on two lines, bigger
            cut = min(range(1, len(ws)), key=lambda k: abs(len(" ".join(ws[:k])) - len(" ".join(ws[k:]))))
            lines = [" ".join(ws[:cut]), " ".join(ws[cut:])]
            size = min(104, min(c.fit_size(ln, "display", 950, 420, 0.3) for ln in lines))
        track = 0.12 + 0.16 * u
        a = fx.smooth(t, self.tc_t + 0.02, 0.32)
        blur = 7 * (1 - fx.expo(t, self.tc_t, 0.45))
        lh = size * 1.12
        cy = c.H / 2 - 20 - lh * (len(lines) - 1) / 2
        for k, ln in enumerate(lines):
            c.put_text(img, ln, "display", size, c.W / 2, cy + k * lh, (240, 236, 228), 420, track, "mm", a,
                       1.0 + 0.035 * u, blur)
        cy += lh * (len(lines) - 1)
        # hairline grows out from the centre, then the sub line resolves under it
        e = fx.expo(t, self.tc_t + 0.12, 0.7)
        hw = 210 * e
        if hw > 1:
            img[int(cy + size * 0.62):int(cy + size * 0.62) + 2, int(c.W / 2 - hw):int(c.W / 2 + hw)] = 200 * a
        sub = b.get("sub", "")
        if sub:
            c.put_text(img, sub.upper(), "grotesk", 30, c.W / 2, cy + size * 0.62 + 46, (170, 170, 168), 300,
                       0.55 + 0.1 * u, "mm", fx.smooth(t, self.tc_t + 0.22, 0.3))
        img = c.bloom(img, 160, 18, 0.35)
        return self.grain(img, i, 4.0)

    def flare_at(self, img, t, t0, dur, x0, x1, y, col, length=1700, thick=7):
        p = c.prog(t, t0, dur)
        if 0 < p < 1:
            e = fx.smooth(t, t0, dur)
            amt = math.sin(math.pi * p) ** 0.8
            x = c.lerp(x0, x1, e)
            fx.flare(img, x, y, length, thick, col, 1.25 * amt)
            fx.flare(img, x, y, 2600, thick * 4, col, 0.22 * amt)   # anamorphic veil
            # little ghost on the opposite side of the frame centre
            gx, gy = c.W - x, c.H - y * 0.9
            r = 46
            ys, xs = fx.grid(2 * r, 2 * r)
            d = np.exp(-(((xs - r) ** 2 + (ys - r) ** 2) / (r * r * 0.45)) ** 2)
            ghost = np.concatenate([d[..., None] * np.float32(col) * 0.6, d[..., None] * 255], -1)
            fx.add(img, ghost, gx - r, gy - r, 0.35 * amt)

    # ---- frame ----
    def frame(self, i, rgb, alpha):
        t = i / self.clip.fps
        B = self.P.beats
        if self.tc_t <= t < self.tc_end:
            return self.title_card(i, t)
        cut = any(0 <= t - ct < 1.0 / self.clip.fps for ct in self.cuts)
        self.cam.blur = 0.0 if cut else 1.0
        img, a = self.cam.place(i, t, rgb, alpha)
        # gate weave (film moving in the gate)
        wx = 1.4 * math.sin(t * 7.1) + 0.8 * math.sin(t * 17.3 + 1.3) + 0.5 * math.sin(i * 2.39)
        wy = 1.1 * math.sin(t * 5.3 + 2.1) + 0.7 * math.sin(t * 13.7) + 0.4 * math.sin(i * 1.71 + 0.5)
        M = cv2.getRotationMatrix2D((c.W / 2, c.H / 2), 0.06 * math.sin(t * 3.3), 1.0)
        M[:, 2] += (wx, wy)
        img = cv2.warpAffine(img, M, (c.W, c.H), borderMode=cv2.BORDER_REFLECT)
        a = cv2.warpAffine(a, M, (c.W, c.H))
        # rack focus on the big word: he drifts soft, then focus pulls onto him as the wall melts away
        if t >= self.dof_t - 0.4:
            sf = 6.5 * math.sin(math.pi * c.prog(t, self.dof_t - 0.4, 0.6))
            sb = 10 * fx.smooth(t, self.dof_t - 0.12, 0.42)
            if sf > 0.4 or sb > 0.4:
                af = cv2.GaussianBlur(a, (0, 0), 2)[..., None]
                fg = cv2.GaussianBlur(img, (0, 0), sf) if sf > 0.4 else img
                bg = cv2.GaussianBlur(img, (0, 0), sb) if sb > 0.4 else img
                img = fg * af + bg * (1 - af)
        img = self.grade(img, a)
        # halation: warm bleed around highlights
        hi = cv2.GaussianBlur(np.clip(img - 175, 0, None), (0, 0), 16)
        img = img + hi * np.float32([0.55, 0.26, 0.12])
        fcx, fcy = self.fr.face_box(i)[:2]
        # anamorphic flares, each one different
        k1 = B[0]["key_t"]
        self.flare_at(img, t, max(0.04, k1 - 0.36), 0.8, -300, c.W + 300, fcy - 60, BLUE_FL)
        if self.tc_end <= t < self.tc_end + 0.5:  # flare pops on the cut back
            amt = math.exp(-(t - self.tc_end) / 0.13)
            fx.flare(img, c.W * 0.78, fcy + 120, 2000, 9, (150, 200, 255), 1.6 * amt)
            fx.flare(img, c.W * 0.78, fcy + 120, 2800, 34, (150, 200, 255), 0.3 * amt)
        if self.dof_t < self.dur:
            self.flare_at(img, t, self.dof_t - 0.2, 0.8, c.W + 300, -300, fcy + 330, AMBER_FL, 1500, 6)
        # light leak once, washing over the return from the title card
        p = c.prog(t, self.tc_end - 0.04, 0.7)
        if 0 < p < 1:
            env = math.sin(math.pi * p) ** 1.3
            lk = leak_layer(p) * (0.6 * env)
            img = 255 - (255 - np.clip(img, 0, 255)) * (1 - np.clip(lk / 255, 0, 1))
        img = c.vignette(img, 0.42)
        img = self.grain(img, i, 8.0)
        # flash-frame open
        fl = 1 - fx.expo(t, 0.0, 0.38)
        if fl > 0.01:
            img = img + (np.float32([255, 250, 242]) - img) * fl
        # letterbox + type in the bars
        bh = self.bars(t)
        if bh > 0.5:
            hb = int(math.floor(bh))
            fr_ = bh - hb
            img[:hb] = 2
            img[c.H - hb:] = 2
            img[hb] = img[hb] * (1 - fr_) + 2 * fr_          # anti-aliased bar edges
            img[c.H - hb - 1] = img[c.H - hb - 1] * (1 - fr_) + 2 * fr_
        self.bar_type(img, t, bh)
        return img

    def bar_type(self, img, t, bh):
        B = self.P.beats
        P = self.P
        # kicker in the top bar during the first shot
        if P.kicker and t < self.tc_t:
            op = fx.smooth(t, 0.25, 0.4) * (1 - fx.smooth(t, self.tc_t - 0.2, 0.15))
            c.put_text(img, P.kicker.upper(), "grotesk", 30, c.W / 2, bh / 2, (170, 178, 182), 420,
                       0.5 + 0.12 * c.prog(t, 0.2, 1.5), "mm", op)
        for k, b in enumerate(B):
            if k in (self.tc, self.fin):
                continue
            t0 = b["key_t"] - 0.05
            t1 = B[k + 1]["key_t"] - 0.08 if k + 1 < len(B) else self.dur + 1
            if k + 1 == self.tc:
                t1 = self.tc_t
            if not (t0 <= t < t1):
                continue
            u = c.prog(t, t0, 1.6)
            op = fx.smooth(t, t0, 0.3) * (1 - fx.smooth(t, t1 - 0.12, 0.12))
            c.put_text(img, b["label"].upper(), "grotesk", 56, c.W / 2, c.H - bh / 2, (240, 236, 228), 300,
                       0.28 + 0.14 * u, "mm", op, 1.0, 4 * (1 - fx.expo(t, t0, 0.4)))
        if self.fin is not None and t >= self.fin_t:
            b = B[self.fin]
            u = c.prog(t, self.fin_t, 1.4)
            lab = b["label"].upper()
            size = min(110, c.fit_size(lab, "display", 940, 420, 0.3))
            yb = c.H - SCOPE / 2 - 20
            a = fx.smooth(t, self.fin_t + 0.12, 0.35)
            c.put_text(img, lab, "display", size, c.W / 2, yb, (244, 238, 228), 420, 0.14 + 0.14 * u, "mm", a,
                       1.0 + 0.03 * u, 6 * (1 - fx.expo(t, self.fin_t + 0.1, 0.5)))
            sub = b.get("sub", "")
            if sub:
                c.put_text(img, sub.upper(), "grotesk", 30, c.W / 2, yb + size * 0.75 + 20, (160, 168, 172), 300,
                           0.55 + 0.1 * u, "mm", fx.smooth(t, self.fin_t + 0.3, 0.35))
            if self.P.title:
                c.put_text(img, self.P.title.upper(), "grotesk", 30, c.W / 2, SCOPE / 2 + 10, (165, 172, 176), 420,
                           0.5, "mm", fx.smooth(t, self.fin_t + 0.35, 0.4))
