"""/news-broadcast: a premium 3D news package. Glossy red, white and navy swoosh bands tear across the frame with
motion blur and reveal the set: the wall turns deep navy and a glowing wireframe globe spins behind the speaker.
A glass lower third builds on the first idea, a red BREAKING slab flips up on a 3D hinge, a light sweep and a
data arc fly on the big beat, the headline cube-rotates to the payoff while the globe sends out a shockwave, with
a live bug, a clock and a ticker running throughout."""
import math

import cv2
import numpy as np

import core as c
import fx
from plan import Plan

W, H = c.W, c.H
NAVY = (8, 20, 58)
NAVY2 = (22, 54, 138)
RED = (226, 30, 52)
GLOW = (92, 170, 255)
WHITE = (255, 255, 255)
GC = (540.0, 600.0)     # globe centre
GR = 452.0              # globe radius
BAR_X, BAR_Y, BAR_W, BAR_H = 56, 1542, 968, 162
TICK_Y, TICK_H = 1722, 66
SW_ANG = math.radians(-15)


def sph(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), -np.sin(la), np.cos(la) * np.sin(lo)], -1).astype(np.float32)


def tight(p):
    """Crop a patch to its visible pixels."""
    ys, xs = np.where(p[..., 3] > 2)
    if len(ys) == 0:
        return p
    return p[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


class Style:
    def __init__(self, clip, opts):
        self.clip = clip
        self.P = P = Plan(clip, opts)
        self.fr = c.Framing(clip, 1215)
        B = self.B = P.beats
        nb = len(B)
        self.T = min(clip.n, len(clip.face)) / clip.fps
        self.lt_t = max(0.36, B[0]["key_t"] - 0.06)
        self.brk_t = B[1]["key_t"] - 0.1 if nb > 2 else self.lt_t + 0.5
        big = B[min(2, nb - 1)] if nb > 2 else B[-1]
        self.big_t = big["key_t"]
        self.pay_t = B[-1]["key_t"] if nb > 1 else self.T * 0.7
        keys = [{"t": 0, "zoom": 1.13, "rot": -2.0}, {"t": 0.0, "zoom": 1.0, "rot": 0.0, "dur": 0.85}]
        if nb > 2:
            keys.append({"t": self.big_t - 0.06, "zoom": 1.085, "ease": "spring", "freq": 1.3, "damp": 0.6})
        keys.append({"t": self.pay_t - 0.04, "zoom": 1.02, "ease": "smooth", "dur": 0.7})
        self.cam = fx.Cam(self.fr, keys, drift=2.5)
        # headline changes: one mode per beat, never the same twice in a row
        modes = ["push", "wipe", "cube"]
        self.heads = [{"t": self.lt_t, "k": 0, "mode": "grow"}]
        for k in range(1, nb):
            mode = "cube" if k == nb - 1 else modes[(k - 1) % 2]
            self.heads.append({"t": B[k]["key_t"] - 0.08, "k": k, "mode": mode})
        # globe geometry
        self.lats = [sph(np.full(91, la), np.linspace(0, 360, 91)) for la in range(-75, 76, 15)]
        self.lons = [sph(np.linspace(-90, 90, 46), np.full(46, lo)) for lo in range(0, 360, 20)]
        rng = np.random.default_rng(5)
        self.cities = sph(rng.uniform(-55, 65, 26), rng.uniform(0, 360, 26))
        a, b = sph(np.float32([38.0]), np.float32([60.0]))[0], sph(np.float32([-12.0]), np.float32([175.0]))[0]
        om = math.acos(float(np.clip(np.dot(a, b), -1, 1)))
        s = np.linspace(0, 1, 80, dtype=np.float32)
        arc = (np.sin((1 - s) * om)[:, None] * a + np.sin(s * om)[:, None] * b) / math.sin(om)
        self.arc = arc * (1 + 0.28 * np.sin(np.pi * s))[:, None]
        tick = "  •  ".join([b["label"].upper() for b in B] + ([P.title.upper()] if P.title else []))
        self.ticker = fx.text(tick + "  •  ", "grotesk", 34, NAVY, 720, 0.02)
        self._tx = {}
        self._head = {}

    def tx(self, s, key="grotesk", size=40, color=WHITE, wght=None, tracking=0.0):
        kk = (s, key, size, color, wght, tracking)
        if kk not in self._tx:
            self._tx[kk] = fx.text(s, key, size, color, wght, tracking)
        return self._tx[kk]

    # ---- the set behind him ----
    def globe(self, t):
        """Wireframe globe layer (premultiplied RGBA, canvas size) with glow, data arc and shockwave."""
        spin = 26 * t + 190 * (1 - fx.expo(t, -0.05, 1.1))
        R = fx.rot3(-22, spin, 8)
        pad = 140
        size = int(2 * (GR * 1.55 + pad))
        ox, oy = GC[0] - size / 2, GC[1] - size / 2
        front = np.zeros((size, size), np.uint8)
        back = np.zeros((size, size), np.uint8)
        hot = np.zeros((size, size), np.uint8)
        sh = 4

        def proj(P, rad=GR):
            Q = P @ R.T
            xy = Q[:, :2] * rad + size / 2
            return (xy * (1 << sh)).astype(np.int32), Q[:, 2]

        for line in self.lats + self.lons:
            xy, z = proj(line)
            fz = z < 0.02
            for mask, sel, wdt in ((front, fz, 2), (back, ~fz, 1)):
                runs = np.split(np.arange(len(z)), np.where(np.diff(sel.astype(int)) != 0)[0] + 1)
                for r in runs:
                    if len(r) > 1 and sel[r[0]]:
                        cv2.polylines(mask, [xy[r]], False, 255, wdt, cv2.LINE_AA, shift=sh)
        xy, z = proj(self.cities)
        for p, zz in zip(xy, z):
            if zz < 0:
                cv2.circle(hot, (int(p[0]), int(p[1])), 5 << sh, 255, -1, cv2.LINE_AA, shift=sh)
        # equator in red
        eq = sph(np.zeros(181), np.linspace(0, 360, 181))
        exy, ez = proj(eq, GR * 1.0)
        red = np.zeros((size, size), np.uint8)
        fz = ez < 0
        runs = np.split(np.arange(len(ez)), np.where(np.diff(fz.astype(int)) != 0)[0] + 1)
        for r in runs:
            if len(r) > 1 and fz[r[0]]:
                cv2.polylines(red, [exy[r]], False, 255, 3, cv2.LINE_AA, shift=sh)
        # data arc on the big beat
        ap = fx.expo(t, self.big_t - 0.04, 0.75)
        if ap > 0:
            n = max(2, int(len(self.arc) * ap))
            axy, az = proj(self.arc[:n])
            cv2.polylines(red, [axy], False, 255, 4, cv2.LINE_AA, shift=sh)
            hx, hy = axy[-1] >> sh
            cv2.circle(hot, (int(hx), int(hy)), 9, 255, -1, cv2.LINE_AA)
            pp = c.prog(t, self.big_t + 0.62, 0.6)
            if 0 < pp < 1:
                cv2.circle(red, (int(hx), int(hy)), int(10 + 70 * c.ease_out_expo(pp)), int(255 * (1 - pp)), 3,
                           cv2.LINE_AA)
        # shockwave on the payoff
        sp = c.prog(t, self.pay_t - 0.02, 0.85)
        if 0 < sp < 1:
            rr = GR * (1.0 + 0.5 * c.ease_out_expo(sp))
            cv2.circle(red, (size // 2, size // 2), int(rr), int(255 * (1 - sp) ** 1.3), int(6 + 10 * (1 - sp)),
                       cv2.LINE_AA)
        f = front.astype(np.float32) / 255
        b = back.astype(np.float32) / 255
        hh = hot.astype(np.float32) / 255
        rd = red.astype(np.float32) / 255
        g1 = cv2.GaussianBlur(f + hh, (0, 0), 5)
        g2 = cv2.GaussianBlur(f * 0.6 + hh + rd, (0, 0), 22)
        col = (f[..., None] * np.float32(GLOW) * 0.95 + b[..., None] * np.float32(GLOW) * 0.22
               + hh[..., None] * 255 + rd[..., None] * np.float32(RED) * 1.1
               + g1[..., None] * np.float32(GLOW) * 0.6 + g2[..., None] * np.float32([120, 150, 255]) * 0.45)
        fade = fx.smooth(t, 0.08, 0.4)
        return col * fade, ox, oy

    # ---- the swoosh reveal ----
    def bands(self, t):
        """Returns (edge position along the sweep axis, list of bands) or None when done."""
        e = fx.smooth(t, -0.27, 0.64)
        if e >= 1:
            return None
        edge = c.lerp(-640, 1640, e)
        return edge, [(0, 34, WHITE, 1.0), (34, 210, RED, 1.0), (210, 330, NAVY2, 1.0), (330, 362, RED, 1.0),
                      (378, 392, WHITE, 0.9), (392, 520, NAVY, 1.0)]

    def swoosh(self, out, slate, t):
        r = self.bands(t)
        if r is None:
            return out
        edge, bands = r
        ys, xs = fx.grid(H, W)
        u = xs * math.cos(SW_ANG) + ys * math.sin(SW_ANG)
        total = bands[-1][1]
        d = edge - u  # distance behind the leading edge
        res = np.where((d < 0)[..., None], slate, out)
        lay = np.zeros((H, W, 4), np.float32)
        for a0, a1, col, op in bands:
            m = np.clip(np.minimum(d - a0, a1 - d) + 0.5, 0, 1) * op
            if m.max() <= 0:
                continue
            gl = np.clip(1 - (d - a0) / max(a1 - a0, 1), 0, 1) ** 2 * 0.35  # gloss at the band's leading edge
            cc = np.float32(col)[None, None] * (1 + gl[..., None]) + 255 * gl[..., None] * 0.4
            lay[..., :3] = lay[..., :3] * (1 - m[..., None]) + cc * m[..., None]
            lay[..., 3] = lay[..., 3] * (1 - m) + 255 * m
        prev = self.bands(t - 0.6 / self.clip.fps)
        v = edge - (prev[0] if prev else edge)
        mv = (v * math.cos(SW_ANG), v * math.sin(SW_ANG))
        lay = fx.dir_blur(lay, *mv)
        return res * (1 - lay[..., 3:4] / 255) + lay[..., :3]

    # ---- broadcast graphics ----
    def headline_patch(self, k):
        if k in self._head:
            return self._head[k]
        b = self.B[k]
        lw = BAR_W - 150 - 48
        lab = b["label"].upper()
        size = min(78, c.fit_size(lab, "anton", lw, None, 0.01))
        p = tight(self.tx(lab, "anton", size, WHITE, None, 0.01))
        sub = tight(self.tx(b["sub"], "grotesk", 28, (196, 210, 244), 640, 0.14)) if b.get("sub") else None
        hh = BAR_H - 16
        out = c.layer(hh, lw + 10)
        gap = 14
        tot = p.shape[0] + (sub.shape[0] + gap if sub is not None else 0)
        y = (hh - tot) / 2
        c.over(out, p, 2, y)
        if sub is not None:
            c.over(out, sub, 4, y + p.shape[0] + gap)
        self._head[k] = out
        return out

    def lower_third(self, img, t):
        g = fx.spring(t, self.lt_t, 1.6, 0.68)
        if g <= 0.002:
            return img
        bw = int(max(30, BAR_W * min(g, 1.04)))
        m = c.rrect(bw, BAR_H, 18)
        fx.glass_shadow(img, m, BAR_X, BAR_Y, blur=24, a=0.45, dy=14)
        gp = fx.glass(img, m, BAR_X, BAR_Y, bevel=22, refract=1.2, magnify=0.0, frost=6.0, chroma=0.35,
                      tint=NAVY, tint_a=0.62, spec=0.7, rim=0.6)
        sw = c.prog(t, self.big_t - 0.02, 0.7)  # the light sweep (once, on the big beat)
        if 0 < sw < 1:
            gp = fx.light_sweep(gp, sw, width=0.09, angle=18, strength=0.65)
        c.over(img, gp, BAR_X, BAR_Y)
        # red index block with gloss
        blk = int(min(150, bw))
        rb = c.fill(c.rrect(blk, BAR_H, 18), RED)
        glo = np.clip(1 - np.linspace(0, 1, BAR_H, dtype=np.float32) / 0.5, 0, 1)[:, None] ** 2 * 0.25
        rb[..., :3] += glo[..., None] * rb[..., 3:4]
        k = max([n for n, hd in enumerate(self.heads) if hd["t"] <= t] or [0])
        num = self.tx(f"{self.heads[k]['k'] + 1:02d}", "anton", 70, WHITE)
        c.over(img, rb, BAR_X, BAR_Y)
        c.over(img, num, BAR_X + blk / 2 - num.shape[1] / 2, BAR_Y + BAR_H / 2 - num.shape[0] / 2,
               c.clamp((g - 0.4) / 0.4))
        # headline area
        hx, hy = BAR_X + 150 + 24, BAR_Y + 8
        hw, hh = BAR_W - 150 - 38, BAR_H - 16
        area = c.layer(hh, hw)
        hd = self.heads[k]
        new = self.headline_patch(hd["k"])
        old = self.headline_patch(self.heads[k - 1]["k"]) if k else None
        if hd["mode"] == "grow":
            e = c.clamp((g - 0.3) / 0.7)
            cut = int(new.shape[1] * fx.smooth(e, 0, 1))
            if cut > 1:
                c.over(area, new[:, :cut], 0, 0)
        elif hd["mode"] == "push":
            e = fx.spring(t, hd["t"], 1.9, 0.7)
            c.over(area, old, 0, -hh * min(e, 1.2), c.clamp(1 - e))
            c.over(area, new, 0, hh * (1 - e))
        elif hd["mode"] == "wipe":
            e = fx.smooth(t, hd["t"], 0.42)
            x = int(hw * e)
            c.over(area, old[:, x:], x, 0)
            c.over(area, new[:, :x], 0, 0)
            if 0 < e < 1:
                c.over(area, c.fill(np.ones((hh, 8), np.float32), RED), x - 4, 0)
        else:  # cube
            e = fx.spring(t, hd["t"], 1.5, 0.62)
            th = 90 * e
            d = hh / 2
            for patch, ang in ((old, th), (new, th - 90)):
                if -89 < ang < 89:
                    a = math.radians(ang)
                    cy = hh / 2 - d * math.sin(a)
                    z = d - d * math.cos(a)
                    fx.card3d(area, patch, patch.shape[1] / 2, cy, rx=-ang, z=z, persp=900, shade=0.6)
        c.over(img, area, hx, hy)
        return img

    def breaking(self, img, t):
        e = fx.spring(t, self.brk_t, 1.5, 0.45)
        if e <= 0.002:
            return
        lab = self.tx("BREAKING", "archivo", 40, WHITE, None, 0.12)
        w, h = lab.shape[1] + 70, 66
        p = c.fill(c.rrect(w, h, 12), RED)
        glo = np.clip(1 - np.linspace(0, 1, h, dtype=np.float32) / 0.55, 0, 1)[:, None] ** 2 * 0.3
        p[..., :3] += glo[..., None] * p[..., 3:4]
        dot = 0.5 + 0.5 * math.cos(t * 2 * math.pi * 1.4)
        c.over(p, c.fill(c.rrect(14, 14, 7), WHITE, 0.45 + 0.55 * dot), 17, h / 2 - 7)
        c.over(p, lab, 44, h / 2 - lab.shape[0] / 2 - 1)
        ang = -95 * (1 - e)  # hinge on the bottom edge
        a = math.radians(ang)
        hinge = BAR_Y - 2
        cy = hinge - (h / 2) * math.cos(a)
        z = -(h / 2) * math.sin(a)
        fx.card3d(img, p, BAR_X + w / 2, cy, rx=-ang, z=z, persp=1100, shade=0.5, shadow=0.35)

    def bug(self, img, t):
        e = fx.spring(t, 0.3, 1.8, 0.7)
        if e <= 0.002:
            return
        x = c.lerp(-260, 56, e)
        live = self.tx("LIVE", "grotesk", 32, WHITE, 800, 0.12)
        lw, lh = live.shape[1] + 62, 58
        p = c.fill(c.rrect(lw, lh, 12), RED)
        pulse = 0.35 + 0.65 * (0.5 + 0.5 * math.cos(t * 2 * math.pi * 1.2))
        c.over(p, c.fill(c.rrect(14, 14, 7), WHITE, pulse), 18, lh / 2 - 7)
        c.over(p, live, 42, lh / 2 - live.shape[0] / 2 - 1)
        mv = (fx.vel(lambda tt: (c.lerp(-260, 56, fx.spring(tt, 0.3, 1.8, 0.7)), 0), t)[0], 0)
        fx.put(img, p, x + lw / 2, 108, opacity=c.clamp(e * 3), mv=mv, shadow=0.35, sblur=10, sdy=5)
        secs = 20 * 3600 + 47 * 60 + 12 + int(t)
        clock = f"{secs // 3600:02d}:{secs // 60 % 60:02d}:{secs % 60:02d}"
        ck = self.tx(clock, "mono", 30, WHITE, 600, 0.04)
        cw = ck.shape[1] + 36
        m = c.rrect(cw, lh, 12)
        cx = int(x + lw + 10)
        if cx < W and cx + cw > 0:
            gp = fx.glass(img, m, cx, 108 - lh // 2, bevel=14, refract=1.0, frost=5, chroma=0.3, tint=NAVY,
                          tint_a=0.6, spec=0.6, rim=0.6)
            c.over(img, gp, cx, 108 - lh // 2, c.clamp(e * 3))
            c.over(img, ck, cx + 18, 108 - ck.shape[0] / 2, c.clamp(e * 3))
        tag = (self.P.kicker or self.P.handle or "").upper()
        if tag:
            e2 = fx.spring(t, 0.42, 1.8, 0.7)
            tp = self.tx(tag, "mono", 28, WHITE, 600, 0.16)
            tw = tp.shape[1] + 34
            xr = c.lerp(W + 40, W - 56 - tw, e2)
            c.over(img, c.fill(c.rrect(tw, 50, 10), NAVY, 0.78), xr, 108 - 25, c.clamp(e2 * 3))
            c.over(img, c.fill(np.ones((50, 5), np.float32), RED), xr, 108 - 25, c.clamp(e2 * 3))
            c.over(img, tp, xr + 20, 108 - tp.shape[0] / 2, c.clamp(e2 * 3))

    def ticker_bar(self, img, t):
        e = fx.spring(t, self.lt_t + 0.12, 1.7, 0.75)
        if e <= 0.002:
            return
        y = TICK_Y + (1 - e) * 140
        strip = c.layer(TICK_H, W)
        strip[..., :3] = 246
        strip[..., 3] = 255
        glo = np.clip(1 - np.linspace(0, 1, TICK_H, dtype=np.float32) / 0.5, 0, 1)[:, None] ** 2 * 10
        strip[..., :3] += glo[..., None]
        tk = self.ticker
        speed = 170
        off = (speed * max(0.0, t - self.lt_t) + 60 * fx.expo(t, self.lt_t, 0.6)) % tk.shape[1]
        x = 240 - off
        while x < W:
            c.over(strip, tk, x, TICK_H / 2 - tk.shape[0] / 2)
            x += tk.shape[1]
        lab = self.tx("TRENDING", "archivo", 30, WHITE, None, 0.1)
        box = c.fill(np.ones((TICK_H, lab.shape[1] + 56), np.float32), NAVY)
        c.over(box, c.fill(np.ones((TICK_H, 6), np.float32), RED), box.shape[1] - 6, 0)
        c.over(box, lab, 26, TICK_H / 2 - lab.shape[0] / 2)
        c.over(strip, box, 0, 0)
        c.over(img, strip, 0, y)

    # ---- frame ----
    def frame(self, i, rgb, alpha):
        t = i / self.clip.fps
        img, a = self.cam.place(i, t, rgb, alpha)
        img = c.grade(img, contrast=1.08, sat=1.02, warm=0.004)
        a = cv2.GaussianBlur(a, (0, 0), 1.4)
        # the set: wall re-lit deep navy, globe behind him
        lum = (img @ np.float32([0.299, 0.587, 0.114]) / 255)[..., None]
        ys = np.linspace(0, 1, H, dtype=np.float32)[:, None, None]
        navy = (np.float32([6, 14, 40]) + (np.float32([34, 74, 168]) - np.float32([6, 14, 40]))
                * np.clip(lum * 1.35, 0, 1) ** 1.1) * (1.05 - 0.25 * ys)
        k = 0.9 * fx.smooth(t, -0.1, 0.25)
        bg = img * (1 - k) + navy * k
        col, ox, oy = self.globe(t)
        gh, gw = col.shape[:2]
        x0, y0 = int(ox), int(oy)
        sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x0 + gw, W), min(y0 + gh, H)
        bg[sy0:sy1, sx0:sx1] += col[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0]
        out = img * a[..., None] + bg * (1 - a[..., None])
        out = c.vignette(out, 0.28)
        # graphics
        self.bug(out, t)
        self.lower_third(out, t)
        self.breaking(out, t)
        self.ticker_bar(out, t)
        # the opening swoosh: bands tear across and reveal the set
        if self.bands(t) is not None:
            slate = fx.gradient(H, W, (10, 26, 74), (4, 9, 28), 70)
            out = self.swoosh(out, slate, t)
        out = c.bloom(out, 210, 22, 0.25)
        return c.grain(out, i, 2.2)
