"""/explainer-panel-reel: a split-screen explainer. The full-frame video springs down into a rounded video card
with motion blur and reveals a clean dark panel above it. On each idea a liquid-glass node with its number pops
in, a glowing connector draws itself to it with a comet head, its label slides out from behind a mask and the
progress rail fills to the next step; energy keeps flowing along the finished connectors and the payoff node
lights up and pulses."""
import math

import cv2
import numpy as np

import core as c
import fx
from plan import Plan

W, H = c.W, c.H
C1 = np.float32([84, 168, 255])     # blue
C2 = np.float32([168, 112, 255])    # violet
WARM = np.float32([255, 196, 120])
CARD = (60, 968, 960, 892)          # x, y, w, h of the video card
PANEL_H = 940
NR = 108                            # node radius


def bez(p0, p1, p2, p3, n=120):
    s = np.linspace(0, 1, n, dtype=np.float32)[:, None]
    return ((1 - s) ** 3) * p0 + 3 * ((1 - s) ** 2) * s * p1 + 3 * (1 - s) * s * s * p2 + s ** 3 * p3


def tight(p):
    ys, xs = np.where(p[..., 3] > 2)
    return p if len(ys) == 0 else p[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


class Style:
    def __init__(self, clip, opts):
        self.clip = clip
        self.P = P = Plan(clip, opts)
        self.fr = c.Framing(clip, 1215)
        B = self.B = P.beats
        nb = self.nb = len(B)
        self.T = min(clip.n, len(clip.face)) / clip.fps
        self.pay_t = B[-1]["key_t"]
        self.cam = fx.Cam(self.fr, [{"t": 0, "zoom": 1.0},
                                    {"t": self.pay_t - 0.05, "zoom": 1.07, "ease": "smooth", "dur": 0.8}], drift=2.5)
        # card crop: full width, centred on the face
        fy = float(np.median([self.fr.face_box(i)[1] for i in range(0, min(clip.n, len(clip.face)), 5)]))
        ch = W * CARD[3] / CARD[2]
        self.crop_cy = c.clamp(fy - 0.40 * ch, 0, H - ch) + ch / 2
        # node layout: zig-zag down the panel
        y0, y1 = 318, 826
        xl, xr = 60 + NR + 6, W - 60 - NR - 6
        self.nodes = []
        for k, b in enumerate(B):
            y = y0 + (y1 - y0) * (k / max(nb - 1, 1))
            left = k % 2 == 0
            self.nodes.append({"k": k, "x": float(xl if left else xr), "y": y, "left": left, "t": b["key_t"]})
        self.paths = []
        ca, sa = math.cos(math.radians(38)), math.sin(math.radians(38))
        for a, b in zip(self.nodes[:-1], self.nodes[1:]):
            sgn = 1 if a["left"] else -1  # leave from the lower side facing the next node, arrive on its upper side
            p0 = np.float32([a["x"] + sgn * (NR + 8) * ca, a["y"] + (NR + 8) * sa])
            p3 = np.float32([b["x"] - sgn * (NR + 8) * ca, b["y"] - (NR + 8) * sa])
            dx = p3[0] - p0[0]
            p1 = p0 + np.float32([dx * 0.5, 34])
            p2 = p3 - np.float32([dx * 0.5, 34])
            pts = bez(p0, p1, p2, p3)
            seg = np.hypot(*np.diff(pts, axis=0).T)
            self.paths.append({"pts": pts, "cum": np.r_[0, np.cumsum(seg)] / seg.sum(), "t1": b["t"] + 0.02})
        self.bg = self._bg()
        self.grad = fx.gradient(PANEL_H, W, C1, C2, 0)
        self._tx = {}
        self._lab = {}

    # ---- static panel ----
    def _bg(self):
        bg = fx.gradient(H, W, (16, 17, 26), (7, 7, 11), 90)
        ys, xs = fx.grid(H, W)
        d = np.hypot((xs - 900) / 620, (ys - 160) / 520)
        bg += (np.exp(-d * d * 1.8) * 0.5)[..., None] * C2 * 0.35
        d = np.hypot((xs - 120) / 600, (ys - 700) / 520)
        bg += (np.exp(-d * d * 1.8) * 0.5)[..., None] * C1 * 0.28
        # blueprint texture: fine grid with brighter crossings, fading out toward the video card
        fade = np.clip((PANEL_H + 10 - ys) / 140, 0, 1) * (0.55 + 0.45 * np.clip(1 - ys / PANEL_H, 0, 1))
        lines = ((np.mod(xs - 6, 54) < 1.2) | (np.mod(ys - 6, 54) < 1.2)).astype(np.float32)
        dots = ((np.mod(xs - 6, 54) < 3.2) & (np.mod(ys - 6, 54) < 3.2)).astype(np.float32)
        tex = cv2.GaussianBlur(lines * 0.5 + dots, (0, 0), 0.6) * fade
        bg += tex[..., None] * np.float32([20, 24, 40])
        d = np.hypot((xs - 540) / 700, (ys - 520) / 420)
        bg += (np.exp(-d * d * 1.5) * fade)[..., None] * np.float32([18, 20, 40])
        rng = np.random.default_rng(9)
        return (bg + rng.normal(0, 1.0, bg.shape)).astype(np.float32)

    def tx(self, s, key="grotesk", size=40, color=(255, 255, 255), wght=None, tracking=0.0):
        kk = (s, key, size, color, wght, tracking)
        if kk not in self._tx:
            self._tx[kk] = fx.text(s, key, size, color, wght, tracking)
        return self._tx[kk]

    def label(self, k):
        if k in self._lab:
            return self._lab[k]
        b = self.B[k]
        last = k == self.nb - 1
        size = 76 if last else 66
        lab = tight(self.tx(b["label"], "grotesk", size, (255, 255, 255), 760, -0.01))
        if lab.shape[1] > 640:
            lab = tight(self.tx(b["label"], "grotesk", max(52, int(size * 640 / lab.shape[1])), (255, 255, 255), 760,
                                -0.01))
        sub = tight(self.tx(b["sub"], "mono", 30, tuple(int(v) for v in (WARM if last else C1 * 0.5 + 128)), 600,
                            0.16)) if b.get("sub") else None
        self._lab[k] = (lab, sub)
        return self._lab[k]

    # ---- pieces ----
    def connectors(self, img, t):
        m = np.zeros((PANEL_H, W), np.uint8)
        hot = np.zeros((PANEL_H, W), np.uint8)
        sh = 4
        heads = []
        for n, p in enumerate(self.paths):
            d = fx.smooth(t, p["t1"] - 0.36, 0.38)
            if d <= 0:
                continue
            pts, cum = p["pts"], p["cum"]
            j = int(np.searchsorted(cum, d))
            q = pts[:max(2, j + 1)]
            cv2.polylines(m, [(q * (1 << sh)).astype(np.int32)], False, 255, 12, cv2.LINE_AA, shift=sh)
            if d < 1:
                heads.append(q[-1])
            else:  # energy flowing along the finished connector
                ph = ((t - p["t1"]) * 0.9 + n * 0.37) % 1.0
                j0, j1 = int(np.searchsorted(cum, max(0, ph - 0.09))), int(np.searchsorted(cum, ph))
                if j1 - j0 > 1:
                    cv2.polylines(hot, [(pts[j0:j1] * (1 << sh)).astype(np.int32)], False, 255, 13, cv2.LINE_AA,
                                  shift=sh)
        if m.max() == 0:
            return
        f = m.astype(np.float32) / 255
        hf = hot.astype(np.float32) / 255
        for hx, hy in heads:
            cv2.circle(hf, (int(hx), int(hy)), 15, 1.0, -1, cv2.LINE_AA)
        g = cv2.GaussianBlur(f * 0.8 + hf, (0, 0), 13)
        g2 = cv2.GaussianBlur(f * 0.5 + hf, (0, 0), 34)
        col = (self.grad * (f * 0.95 + g * 1.5 + g2 * 0.9)[..., None] + 255 * (hf * 0.9 + g2 * hf.max() * 0.4)[..., None])
        img[:PANEL_H] += col

    def node(self, img, nd, t, active):
        k = nd["k"]
        last = k == self.nb - 1
        e = fx.spring(t, nd["t"] - 0.04, 2.1, 0.42)
        if e <= 0.01:
            return
        pulse = 0.0
        if last:
            pulse = 0.5 + 0.5 * math.sin((t - nd["t"]) * 2 * math.pi * 1.4 - math.pi / 2)
            pulse *= fx.smooth(t, nd["t"] + 0.25, 0.3)
        r = NR * e * (1 + 0.06 * pulse)
        x, y = nd["x"], nd["y"]
        # glowing orb under the glass (the active / payoff node)
        if active > 0.01:
            R = int(r * 2.4) + 4
            ys, xs = fx.grid(2 * R, 2 * R)
            dd = np.hypot(xs - R, ys - R)
            col = (WARM * 0.6 + C2 * 0.4) if last else C1 * (1 - (ys / (2 * R)))[..., None] + C2 * (ys / (2 * R))[..., None]
            core = np.clip(1 - dd / (r * 0.86), 0, 1) ** 0.6
            halo = np.exp(-(dd / (r * 1.25)) ** 2) * 0.55
            m = (core + halo) * active * (1 + 0.5 * pulse)
            fx.add(img, np.concatenate([m[..., None] * (col if col.ndim == 3 else col[None, None]),
                                        m[..., None] * 255], -1), x - R, y - R, 1.0)
        D = int(2 * r) + 2
        if D < 8:
            return
        msk = c.rrect(D, D, D // 2)
        x0, y0 = int(x - D / 2), int(y - D / 2)
        fx.glass_shadow(img, msk, x0, y0, blur=16, a=0.4, dy=10)
        c.over(img, fx.glass(img, msk, x0, y0, bevel=max(8, D * 0.3), refract=1.5, magnify=0.12, frost=1.2,
                             chroma=0.7, tint=(30, 40, 80), tint_a=0.22, spec=1.0, rim=0.9), x0, y0)
        num = self.tx(f"{k + 1:02d}", "grotesk", 86, (255, 255, 255), 720)
        fx.put(img, num, x, y, min(1.0, e), opacity=c.clamp((e - 0.3) / 0.4))
        if last:  # payoff rings
            for ph in (0.0, 0.5):
                q = ((t - nd["t"] - 0.2) * 0.9 + ph) % 1.0 if t > nd["t"] + 0.2 else -1
                if q < 0:
                    continue
                rr = r * (1.08 + 0.62 * q)
                RR = int(rr) + 8
                ys, xs = fx.grid(2 * RR, 2 * RR)
                ring = np.exp(-((np.hypot(xs - RR, ys - RR) - rr) / 3.0) ** 2) * (1 - q) ** 1.6 * 0.9
                fx.add(img, c.fill(ring, tuple(int(v) for v in WARM)), x - RR, y - RR)

    def node_label(self, img, nd, t):
        k = nd["k"]
        e = fx.expo(t, nd["t"] + 0.02, 0.55)
        if e <= 0.005:
            return
        lab, sub = self.label(k)
        gap = 30
        for n, p in enumerate((lab, sub)):
            if p is None:
                continue
            ee = fx.expo(t, nd["t"] + 0.02 + 0.09 * n, 0.55)
            if ee <= 0.005:
                continue
            h, w = p.shape[:2]
            y = nd["y"] - (lab.shape[0] + (sub.shape[0] + 16 if sub is not None else 0)) / 2
            if n:
                y += lab.shape[0] + 16
            vis = int(w * ee)  # mask reveal from the node side, text slides out with it
            if vis < 2:
                continue
            if nd["left"]:  # text slides right out from behind a mask at the node's edge
                c.over(img, p[:, w - vis:], nd["x"] + NR + gap, y)
            else:           # ... or left
                c.over(img, p[:, :vis], nd["x"] - NR - gap - vis, y)

    def header(self, img, t):
        e = fx.expo(t, 0.22, 0.6)
        if e <= 0.005:
            return
        k, _, _ = self.P.at(t)
        kick = (self.P.kicker or self.P.title or "").upper()
        dy = (1 - e) * -30
        if kick:
            kp = self.tx(kick, "mono", 30, (206, 212, 232), 560, 0.18)
            c.over(img, kp, 60, 78 + dy, e)
        steps = sum(1 for nd in self.nodes if t >= nd["t"] - 0.04)
        st = self.tx(f"STEP {max(steps, 1):02d}/{self.nb:02d}", "mono", 30, (255, 255, 255), 640, 0.12)
        c.over(img, st, W - 60 - st.shape[1], 78 + dy, e)
        # progress rail
        y = 150
        rw = W - 120
        c.over(img, c.fill(c.rrect(rw, 8, 4), (255, 255, 255), 0.12 * e), 60, y)
        fill = 0.0
        for n, nd in enumerate(self.nodes):
            fill += (1 / self.nb) * fx.spring(t, nd["t"] - 0.02, 1.6, 0.7)
        fill = c.clamp(fill, 0, 1.02)
        fw = int(rw * min(fill, 1.0))
        if fw > 8:
            bar = np.concatenate([fx.gradient(8, fw, C1, C2, 0) * c.rrect(fw, 8, 4)[..., None],
                                  c.rrect(fw, 8, 4)[..., None] * 255], -1)
            c.over(img, fx.glow(bar, 8, 0.9), 60 - 20, y - 20, e)
        for n in range(self.nb + 1):
            xx = 60 + rw * n / self.nb
            on = fill >= n / self.nb - 1e-3
            cv2.circle(img, (int(xx), y + 4), 7, (255, 255, 255) if on else (90, 92, 110), -1, cv2.LINE_AA)

    def card(self, img, F, t):
        e = fx.spring(t, 0.02, 1.55, 0.6)
        x, y, w, h = CARD
        dx, dy, dw, dh = c.lerp(0, x, e), c.lerp(0, y, e), c.lerp(W, w, e), c.lerp(H, h, e)
        r = c.lerp(0, 44, min(e, 1.0))
        sc = c.lerp(1.0, w / W, e)
        cw, chh = dw / sc, dh / sc
        ccy = c.lerp(H / 2, self.crop_cy, e)
        y0 = c.clamp(ccy - chh / 2, 0, max(0.0, H - chh))
        x0 = c.clamp(W / 2 - cw / 2, 0, max(0.0, W - cw))
        M = np.float32([[sc, 0, -x0 * sc], [0, sc, -y0 * sc]])
        iw, ih = max(4, int(round(dw))), max(4, int(round(dh)))
        crop = cv2.warpAffine(F, M, (iw, ih), flags=cv2.INTER_AREA if sc < 0.95 else cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
        m = c.rrect(iw, ih, max(1, int(r)))[..., None]
        p = np.concatenate([crop * m, m * 255], -1)
        if e > 0.3:  # glass edge
            ring = np.clip(c.rrect(iw, ih, max(1, int(r))) - np.pad(c.rrect(iw - 4, ih - 4, max(1, int(r) - 2)), 2), 0, 1)
            c.over(p, c.fill(ring, (255, 255, 255), 0.22 * c.clamp((e - 0.3) / 0.4)))
        p0 = fx.spring(t - 0.6 / self.clip.fps, 0.02, 1.55, 0.6)
        mvy = (c.lerp(0, y, e) + c.lerp(H, h, e) / 2) - (c.lerp(0, y, p0) + c.lerp(H, h, p0) / 2)
        fx.put(img, p, dx + iw / 2, dy + ih / 2, opacity=1.0, mv=(0, mvy), shadow=0.55 * min(e, 1.0), sblur=34, sdy=20)
        # name chip on the card
        name = self.P.name or self.P.handle
        if name:
            ce = fx.expo(t, 0.62, 0.5)
            if ce > 0.01:
                nt = self.tx(name, "grotesk", 30, (255, 255, 255), 640)
                cw_, chh_ = nt.shape[1] + 62, 58
                mx, my = int(x + 26), int(y + 26)
                msk = c.rrect(cw_, chh_, chh_ // 2)
                gp = fx.glass(img, msk, mx, my, bevel=16, refract=1.2, frost=6, chroma=0.4, tint=(16, 18, 28),
                              tint_a=0.45, spec=0.8, rim=0.7)
                vis = int(cw_ * ce)
                c.over(img, gp[:, :vis], mx, my)
                cv2.circle(img, (mx + 26, my + chh_ // 2), 7, (255, 80, 80), -1, cv2.LINE_AA)
                c.over(img, nt[:, :max(1, vis - 44)], mx + 44, my + chh_ / 2 - nt.shape[0] / 2 - 1)

    # ---- frame ----
    def frame(self, i, rgb, alpha):
        t = i / self.clip.fps
        F, _ = self.cam.place(i, t, rgb, alpha)
        F = c.grade(F, contrast=1.08, sat=1.04, warm=0.01)
        img = self.bg.copy()
        drift = 0.5 + 0.5 * math.sin(t * 0.9)
        img[:PANEL_H] += self.grad * (0.035 + 0.02 * drift)
        self.connectors(img, t)
        cur = max([n for n, nd in enumerate(self.nodes) if t >= nd["t"] - 0.04] or [-1])
        for n, nd in enumerate(self.nodes):
            if n > cur:
                continue
            nxt = self.nodes[n + 1]["t"] if n + 1 < self.nb else 99
            act = fx.smooth(t, nd["t"] - 0.04, 0.2) * (1 - fx.smooth(t, nxt - 0.04, 0.3))
            if n == self.nb - 1:
                act = fx.smooth(t, nd["t"] - 0.04, 0.2)
            self.node(img, nd, t, act)
            self.node_label(img, nd, t)
        self.header(img, t)
        self.card(img, F, t)
        img = c.bloom(img, 205, 22, 0.25)
        img = c.vignette(img, 0.25)
        return c.grain(img, i, 2.2)
