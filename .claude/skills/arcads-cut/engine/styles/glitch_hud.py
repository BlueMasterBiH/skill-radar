"""/glitch-hud: the clip seen through a cyber targeting system. A full-frame glitch burst tears open into the shot as face-tracking brackets spring in from the corners and lock on with a live target readout; desaturated image, scanlines, a viewfinder frame and scrolling data columns; each idea decodes out of random characters into a mono label, and every beat lands its own glitch: slice displacement with RGB split, a scan tear that rips down the frame while the brackets re-acquire, and a mosaic that resolves into a green match."""
import math

import cv2
import numpy as np

import core as c
import fx
from plan import Plan

CYAN = (0, 236, 255)
GREEN = (90, 255, 160)
MAGENTA = (255, 40, 120)
GLYPHS = "#$%&*+=?@ABCDEFGHJKLMNPQRSTUVWXYZ0123456789<>/\\[]"
LAB_Y = 1530


def rng_for(i, salt):
    return np.random.default_rng(i * 7919 + salt * 104729 + 17)


def slice_glitch(img, i, amt, salt, density=0.55):
    """Horizontal slice displacement with per-slice colour offsets (deterministic per frame)."""
    if amt < 1:
        return img
    r = rng_for(i, salt)
    out = img.copy()
    y = 0
    while y < c.H:
        h = int(r.integers(6, 110))
        if r.random() < density:
            dx = int(r.normal(0, 1) * amt)
            out[y:y + h] = np.roll(img[y:y + h], dx, axis=1)
            if r.random() < 0.6:
                ch = int(r.integers(0, 3))
                out[y:y + h, :, ch] = np.roll(img[y:y + h, :, ch], dx + int(r.normal(0, 0.35) * amt), axis=1)
        y += h
    return out


def blocks(img, i, n, salt):
    """Corrupted blocks: displaced copies, inverted tiles and colour bars."""
    r = rng_for(i, salt + 5)
    out = img
    for _ in range(n):
        w, h = int(r.integers(80, 520)), int(r.integers(14, 140))
        x, y = int(r.integers(0, c.W - 40)), int(r.integers(0, c.H - 20))
        w, h = min(w, c.W - x), min(h, c.H - y)
        kind = r.random()
        if kind < 0.45:
            sx = int(np.clip(x + r.normal(0, 160), 0, c.W - w))
            sy = int(np.clip(y + r.normal(0, 60), 0, c.H - h))
            out[y:y + h, x:x + w] = img[sy:sy + h, sx:sx + w].copy()
        elif kind < 0.7:
            out[y:y + h, x:x + w] = 255 - out[y:y + h, x:x + w]
        else:
            col = np.float32(CYAN if r.random() < 0.6 else MAGENTA)
            out[y:y + h, x:x + w] = out[y:y + h, x:x + w] * 0.35 + col * 0.65
    return out


class Style:
    def __init__(self, clip, opts):
        self.clip = clip
        self.P = P = Plan(clip, opts)
        self.fr = c.Framing(clip, 1215)
        self.N = min(clip.n, len(clip.face))
        self.dur = self.N / clip.fps
        B = P.beats
        self.lock_t = 0.48
        # one glitch per beat after the first: slices, scan tear, mosaic (the payoff always resolves as a mosaic)
        kinds = ["slice", "tear", "mosaic"]
        self.glitch = {}
        for k in range(1, len(B)):
            self.glitch[k] = "mosaic" if k == len(B) - 1 else kinds[(k - 1) % 2]
        self.cam = fx.Cam(self.fr, self.camera_keys(), drift=2.0)
        ys = np.arange(c.H)
        self.scan = np.where(ys % 4 < 2, 1.0, 0.8).astype(np.float32)[:, None, None]
        self.hex = {}

    def camera_keys(self):
        B = self.P.beats
        keys = [{"t": 0.0, "zoom": 1.12}, {"t": 0.02, "zoom": 1.0, "ease": "spring", "freq": 1.6, "damp": 0.62}]
        for k, kind in self.glitch.items():
            kt = B[k]["key_t"]
            nx = B[k + 1]["key_t"] - 0.05 if k + 1 < len(B) else 99
            if kind == "slice":
                keys += [{"t": kt, "zoom": 1.08, "dx": -18, "ease": "expo", "dur": 0.14},
                         {"t": min(kt + 0.5, nx), "zoom": 1.04, "dx": 0, "ease": "smooth", "dur": 0.8}]
            elif kind == "tear":
                keys += [{"t": kt, "zoom": 1.14, "ease": "spring", "freq": 2.4, "damp": 0.45},
                         {"t": min(kt + 0.4, nx), "zoom": 1.07, "ease": "smooth", "dur": 0.6}]
            else:
                keys += [{"t": kt, "zoom": 1.02, "dx": 0, "ease": "expo", "dur": 0.4}]
        return keys

    # ---- tracking ----
    def target(self, i, t):
        p = self.cam.at(t)
        fcx, fcy, fw, fh = self.fr.face_box(i)
        z = p["zoom"]
        cx, cy = fcx + p["dx"], fcy + p["dy"] - fh * 0.1 * z
        w, h = fw * z * 1.22, fh * z * 1.02
        # the brackets spring in from the frame corners and lock on
        e = fx.spring(t, 0.04, 1.5, 0.72)
        cx, cy = c.lerp(c.W / 2, cx, e), c.lerp(c.H / 2, cy, e)
        w, h = c.lerp(c.W - 90, w, e), c.lerp(c.H - 90, h, e)
        for k, kind in self.glitch.items():  # re-acquire on the scan tear
            if kind == "tear":
                kt = self.P.beats[k]["key_t"]
                if t >= kt:
                    s = 1 + 0.7 * (1 - fx.spring(t, kt, 2.0, 0.5))
                    w, h = w * s, h * s
        return cx, cy, w, h

    # ---- HUD drawing ----
    def hud(self, i, t):
        B = self.P.beats
        H8 = np.zeros((c.H, c.W, 3), np.uint8)
        T = np.zeros((c.H, c.W, 3), np.float32)
        cx, cy, w, h = self.target(i, t)
        locked = t >= self.lock_t
        mos = [k for k, kd in self.glitch.items() if kd == "mosaic"]
        match = bool(mos) and t >= B[mos[0]]["key_t"] + 0.2
        col = GREEN if match else CYAN
        fl = math.exp(-max(t - self.lock_t, 0) / 0.12) if locked else 0.0
        bc = tuple(int(c.lerp(v, 255, fl)) for v in col)
        x0, y0, x1, y1 = cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2
        arm = min(w, h) * 0.22
        th = 5 + int(4 * fl)
        for (xa, ya, sx, sy) in ((x0, y0, 1, 1), (x1, y0, -1, 1), (x0, y1, 1, -1), (x1, y1, -1, -1)):
            pts = np.int32([[xa + sx * arm, ya], [xa, ya], [xa, ya + sy * arm]]) * 16
            cv2.polylines(H8, [pts], False, bc, th, cv2.LINE_AA, shift=4)
        # lock ring: the box flashes outward when it locks
        if locked and t < self.lock_t + 0.3:
            q = c.prog(t, self.lock_t, 0.3)
            s = 1 + 0.45 * c.ease_out_cubic(q)
            g = int(255 * (1 - q))
            cv2.rectangle(H8, (int(cx - w * s / 2), int(cy - h * s / 2)), (int(cx + w * s / 2), int(cy + h * s / 2)),
                          (g, g, g), 2, cv2.LINE_AA)
        # centre tick marks on the box sides
        for (xa, ya, xb, yb) in ((cx, y0 - 14, cx, y0 + 10), (cx, y1 - 10, cx, y1 + 14),
                                 (x0 - 14, cy, x0 + 10, cy), (x1 - 10, cy, x1 + 14, cy)):
            cv2.line(H8, (int(xa), int(ya)), (int(xb), int(yb)), col, 2, cv2.LINE_AA)
        # target readout with a leader line
        rx, ry = int(min(x1 + 26, c.W - 300)), int(y0 + 6)
        op = fx.smooth(t, 0.12, 0.25)
        if op > 0:
            cv2.line(H8, (int(x1), int(y0)), (rx - 6, ry - 6), col, 2, cv2.LINE_AA)
            if match:
                st = "MATCH  100%"
            elif locked:
                st = "TARGET LOCKED"
            else:
                st = "ACQUIRING" + "." * (i % 4)
            conf = 97.0 + 2.6 * fx.expo(t, 0.1, 0.6) + 0.2 * math.sin(i * 1.7)
            fb = self.fr.face_box(i)
            lines = [(st, col if locked else MAGENTA, 600),
                     ((self.P.handle or self.P.name or "SUBJECT 01").upper(), (230, 240, 240), 500),
                     (f"X {int(fb[0]):04d}  Y {int(fb[1]):04d}", (170, 200, 205), 500),
                     (f"CONF {min(conf, 99.9):04.1f}%", (170, 200, 205), 500)]
            for n, (s, cc, wg) in enumerate(lines):
                if locked or n == 0 or i % 2 == 0:
                    c.put_text(T, s, "mono", 28, rx, ry + 22 + n * 38, cc, wg, 0.04, "ls", op)
        # viewfinder frame + header
        m, L = 40, 70
        for (xa, ya, sx, sy) in ((m, m, 1, 1), (c.W - m, m, -1, 1), (m, c.H - m, 1, -1), (c.W - m, c.H - m, -1, -1)):
            cv2.polylines(H8, [np.int32([[xa + sx * L, ya], [xa, ya], [xa, ya + sy * L]])], False, (150, 200, 205), 2,
                          cv2.LINE_AA)
        if i % 25 < 15:
            cv2.circle(H8, (m + 30, m + 44), 9, MAGENTA, -1, cv2.LINE_AA)
        fs = int(t * 25)
        c.put_text(T, f"REC  TC 00:00:{int(t):02d}:{fs % 25:02d}", "mono", 28, m + 52, m + 54, (210, 230, 232), 500,
                   0.04, "ls")
        c.put_text(T, "AI-VISION", "mono", 28, c.W - m - 18, m + 54, (210, 230, 232), 600, 0.12, "rs")
        # data columns scrolling on both edges
        for side in (0, 1):
            x = m + 4 if side == 0 else c.W - m - 4
            off = t * (46 if side == 0 else -38)
            r0 = int(math.floor(off / 40))
            for n in range(13):
                rr = r0 + n
                yy = 760 + n * 40 - (off - r0 * 40)
                if yy < 740 or yy > 1300:
                    continue
                h_ = (rr * 2654435761 + side * 97) & 0xFFFFFF
                s = f"{h_ & 0xFFFF:04X}" if side == 0 else f"{(h_ >> 8) % 1000:03d}.{h_ % 10}"
                a = 0.5 * (1 - abs(yy - 1020) / 300)
                c.put_text(T, s, "mono", 28, x, yy, GREEN if side == 0 else CYAN, 500, 0.0, "ls" if side == 0 else "rs",
                           max(0.0, a))
        self.label(T, H8, i, t)
        Hf = H8.astype(np.float32) + T
        return Hf

    def label(self, T, H8, i, t):
        B = self.P.beats
        k = max([n for n, b in enumerate(B) if b["key_t"] - 0.06 <= t] or [-1])
        if k < 0:
            return
        b = B[k]
        t0 = b["key_t"] - 0.06
        lab = b["label"].upper()
        size = min(112, c.fit_size(lab + "_", "mono", 900, 700, 0.0))
        dur = 0.36
        r = rng_for(i, 77 + k)
        out = []
        for n, ch in enumerate(lab):
            tr = t0 + 0.04 + dur * n / max(1, len(lab))
            if ch == " " or t >= tr:
                out.append(ch)
            else:
                out.append(GLYPHS[int(r.integers(0, len(GLYPHS)))])
        s = "".join(out)
        done = t >= t0 + 0.04 + dur
        cur = "_" if (i // 6) % 2 == 0 or not done else " "
        x = 70
        op = 0.55 + 0.45 * fx.smooth(t, t0, 0.08)
        col = GREEN if self.glitch.get(k) == "mosaic" and done else CYAN
        # rule that draws across, index + sub line above it
        e = fx.expo(t, t0, 0.5)
        yr = LAB_Y - size - 26
        cv2.line(H8, (x, int(yr)), (int(x + 940 * e), int(yr)), col, 2, cv2.LINE_AA)
        for q in range(0, 11):
            xx = x + q * 94
            if xx <= x + 940 * e:
                cv2.line(H8, (int(xx), int(yr)), (int(xx), int(yr + (12 if q % 5 == 0 else 6))), col, 2, cv2.LINE_AA)
        sub = b.get("sub", "")
        hdr = f"[{k + 1:02d}/{len(B):02d}]" + (f"  {sub.upper()}" if sub else "")
        nshow = int(len(hdr) * fx.smooth(t, t0 + 0.05, 0.3))
        c.put_text(T, hdr[:nshow], "mono", 30, x, yr - 18, (200, 235, 220), 500, 0.06, "ls", op)
        c.put_text(T, s + cur, "mono", size, x - 4, LAB_Y, col, 700, 0.0, "ls", op)

    # ---- frame ----
    def frame(self, i, rgb, alpha):
        t = i / self.clip.fps
        B = self.P.beats
        img, a = self.cam.place(i, t, rgb, alpha)
        img = c.grade(img, contrast=1.14, sat=0.56, gain=0.95)
        img = img * np.float32([0.9, 1.0, 1.02])
        # panel backing under the label for legibility
        img[LAB_Y - 230:LAB_Y + 50, :] *= 0.55
        # beat glitches
        for k, kind in self.glitch.items():
            kt = B[k]["key_t"]
            if kind == "slice":
                q = c.prog(t, kt - 0.08, 0.36)
                if 0 < q < 1:
                    amt = 120 * math.sin(math.pi * q) ** 0.7
                    img = slice_glitch(img, i, amt, 11 + k)
                    img = fx.rgb_shift(img, 22 * math.sin(math.pi * q))
            elif kind == "tear":
                q = c.prog(t, kt - 0.06, 0.46)
                if 0 < q < 1:
                    yc = c.lerp(-200, c.H + 200, c.ease_in_out_cubic(q))
                    y0, y1 = int(max(0, yc - 200)), int(min(c.H, yc + 200))
                    if y1 > y0:
                        band = img[y0:y1].copy()
                        r = rng_for(i, 23)
                        band = np.roll(band, int(r.normal(0, 110)), axis=1)
                        hb = band.shape[0]
                        if hb > 20:  # pixel drag: the lower part of the tear smears one row downwards
                            k0 = int(hb * 0.55)
                            band[k0:] = band[k0:k0 + 1] * 0.6 + band[k0:] * 0.4
                        band = fx.rgb_shift(band, 40)
                        band = band * 1.3 + 20
                        img[y0:y1] = band
                        img[max(0, y0 - 2):y0 + 2] = np.float32(CYAN)
                        img[max(0, y1 - 2):min(c.H, y1 + 2)] = np.float32(MAGENTA)
                    img = fx.rgb_shift(img, 10 * math.sin(math.pi * q))
            else:  # mosaic resolve
                q = c.prog(t, kt - 0.04, 0.42)
                if 0 < q < 1:
                    bs = int(round(c.lerp(64, 1, c.ease_out_cubic(q)) / 4) * 4)
                    if bs >= 4:
                        sm = cv2.resize(img, (c.W // bs, c.H // bs), interpolation=cv2.INTER_AREA)
                        img = cv2.resize(sm, (c.W, c.H), interpolation=cv2.INTER_NEAREST)
                    img = fx.rgb_shift(img, 14 * (1 - q))
        # opening: full-frame glitch burst that resolves into the shot
        ob = 1 - fx.smooth(t, 0.0, 0.5)
        if ob > 0.01:
            img = slice_glitch(img, i, 260 * ob, 3, density=0.75)
            img = blocks(img, i, int(16 * ob) + 2, 3)
            img = fx.rgb_shift(img, 40 * ob, angle=8)
            img = img * (1 + 0.5 * ob * (i % 2 == 0)) + 30 * ob
        # scanlines + rolling bright band
        img = img * self.scan
        yb = (t * 900) % (c.H + 400) - 200
        ys = np.arange(c.H, dtype=np.float32)
        img = img + (np.exp(-((ys - yb) / 60) ** 2) * 14)[:, None, None]
        # HUD on top, glowing
        Hf = self.hud(i, t)
        if ob > 0.01:  # the HUD glitches with the opening burst
            Hf = slice_glitch(Hf, i, 120 * ob, 9)
        g = cv2.GaussianBlur(Hf, (0, 0), 5)
        img = img * (1 - np.clip(Hf.max(2, keepdims=True) / 255, 0, 1) * 0.6) + Hf + g * 0.9
        img = c.vignette(img, 0.4)
        return c.grain(img, i, 4.0)
