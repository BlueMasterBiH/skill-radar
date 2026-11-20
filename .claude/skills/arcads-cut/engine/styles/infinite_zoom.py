"""/infinite-zoom: a Droste tunnel built from the live video. The frame holds a smaller, slightly turned copy of
itself floating behind the speaker (thin white border, soft shadow, viewfinder corners), which holds another, and
another. It opens with a fast motion-blurred pull-out from deep inside the tunnel, the geometry breathes on the
first idea, then on every idea the camera spirals seamlessly into the next frame, each push with its own speed,
and every frame carries its idea as a label that grows into place."""
import math

import cv2
import numpy as np

import core as c
import fx
from plan import Plan

W, H = c.W, c.H
AMBER = (255, 186, 102)
S0 = 0.36          # inner frame scale
R0 = 4.0           # inner frame turn per level (deg)
RAD = 46           # frame corner radius (frame-local px)
DEPTH = 6          # levels drawn inside the base


def imp(t, t0, freq=2.2, decay=5.0):
    """Decaying kick 0 -> +1 -> ... -> 0 from t0 (eased in over the first few ms)."""
    x = t - t0
    if x <= 0:
        return 0.0
    return math.sin(2 * math.pi * freq * x) * math.exp(-decay * x) * c.clamp(x / 0.04)


def io_expo(x):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    return 2 ** (20 * x - 10) / 2 if x < 0.5 else (2 - 2 ** (-20 * x + 10)) / 2


def round_outline(w, h, r, n=8):
    """Closed outline of a rounded rectangle (local px)."""
    pts = []
    for cx, cy, a0 in ((w - r, r, -90), (w - r, h - r, 0), (r, h - r, 90), (r, r, 180)):
        for k in range(n + 1):
            a = math.radians(a0 + 90 * k / n)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    pts.append(pts[0])
    return np.float32(pts)


def zblur(img, cx, cy, amount):
    """Smooth radial zoom blur: three stacked passes (widths 15/30/55 %) instead of one stepped pass, so borders
    and type smear into clean streaks without ghost copies."""
    if amount < 0.004:
        return img
    for w in (0.15, 0.3, 0.55):
        img = fx.zoom_blur(img, cx, cy, amount * w, n=5)
    return img


class Style:
    def __init__(self, clip, opts):
        self.clip = clip
        self.P = P = Plan(clip, opts)
        self.fr = fr = c.Framing(clip, 1215)
        B = self.B = P.beats
        self.nb = nb = len(B)
        fxs = float(np.median([fr.face_box(i)[0] for i in range(0, min(clip.n, len(clip.face)), 5)]))
        side = 1 if fxs <= W / 2 else -1  # inner frame on the roomier side of the face
        self.C1 = np.float32([W / 2 + side * W * 0.273, H * 0.247])
        self.T = min(clip.n, len(clip.face)) / clip.fps
        # zoom programme: opening pull-out, breathe on beat 0, one push per idea (the last one goes two levels)
        self.pushes = []
        for k in range(1, nb):
            b = B[k]
            last = k == nb - 1
            mode = "double" if last else ("smooth" if k % 2 else "expo")
            st = {"smooth": b["key_t"] - 0.30, "expo": b["key_t"] - 0.1, "double": b["key_t"] - 0.36}[mode]
            self.pushes.append({"t": st, "mode": mode, "k": k})
        self.breath_t = B[0]["key_t"] if nb else 0.4
        self.reveal_t = max(0.3, self.breath_t - 0.12)
        self.expo_k = [p["k"] for p in self.pushes if p["mode"] == "expo"][:1]
        # frame index -> label beat
        self.lab_of = {}
        for k in range(nb - 1):
            self.lab_of[k] = k
        if nb > 1:
            self.lab_of[nb] = nb - 1
        else:
            self.lab_of[0] = 0
        self.mask = c.rrect(W, H, RAD)
        self.outline = round_outline(W, H, RAD)
        self._tx = {}
        self._tags = {}

    # ---- timing ----
    def state(self, t):
        # pull-out mostly done by 0.12 s; pushes are whips (in-out expo) so only ~3 frames are fast enough to blur
        u = 4.0 * (1 - fx.expo(t, -0.06, 0.45)) + 0.035 * math.sin(t * 1.3) * fx.smooth(t, 0.5, 0.6)  # + breathing
        for p in self.pushes:
            if p["mode"] == "smooth":
                u += io_expo(c.prog(t, p["t"], 0.7))
            elif p["mode"] == "expo":
                u += fx.expo(t, p["t"], 0.62)
            else:
                u += 2 * io_expo(c.prog(t, p["t"], 0.8))
        kick = imp(t, self.breath_t - 0.04)
        s = S0 * (1 + 0.075 * kick)
        r = R0 + 3.5 * kick
        return u, s, r

    def fixed_point(self, s, r):
        a = math.radians(r)
        R = np.float32([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        C0 = np.float32([W / 2, H / 2])
        A = np.eye(2, dtype=np.float32) - s * R
        return np.linalg.solve(A, self.C1 - s * R @ C0).astype(np.float32)

    @staticmethod
    def affine(Ps, sig, th):
        a = math.radians(th)
        R = np.float32([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]) * sig
        M = np.zeros((2, 3), np.float32)
        M[:, :2] = R
        M[:, 2] = Ps - R @ Ps
        return M

    # ---- type ----
    def tx(self, s, key="grotesk", size=40, color=(255, 255, 255), wght=None, tracking=0.0):
        kk = (s, key, size, color, wght, tracking)
        if kk not in self._tx:
            self._tx[kk] = fx.text(s, key, size, color, wght, tracking)
        return self._tx[kk]

    def tag(self, k):
        if k in self._tags:
            return self._tags[k]
        b = self.B[k]
        lab = self.tx(b["label"], "grotesk", 70, (255, 255, 255), 760, -0.01)
        if lab.shape[1] > 560:
            lab = self.tx(b["label"], "grotesk", int(70 * 560 / lab.shape[1]), (255, 255, 255), 760, -0.01)
        top = self.tx(f"{k + 1:02d} / {self.nb:02d}", "mono", 30, AMBER, 600, 0.12)
        sub = self.tx(b["sub"], "mono", 28, (236, 232, 226), 520, 0.16) if b.get("sub") else None
        small = c.layer(lab.shape[0] + 110, max(lab.shape[1], top.shape[1], sub.shape[1] if sub is not None else 0) + 8)
        c.over(small, top, 2, 0)
        if sub is not None:
            c.over(small, sub, 2, top.shape[0] + lab.shape[0] + 6)
        big = c.layer(*small.shape[:2])
        c.over(big, lab, 0, top.shape[0] - 4)
        sh, off = c.shadow(big, 14, 0.55, 4)
        bigs = c.layer(*sh.shape[:2])
        c.over(bigs, sh, 0, 0)
        c.over(bigs, big, -off[0], -off[1] - 4)
        smallp = c.layer(*bigs.shape[:2])
        c.over(smallp, small, -off[0], -off[1] - 4)
        self._tags[k] = (bigs, smallp, -off[0], -off[1] - 4)
        return self._tags[k]

    # ---- frame ----
    def frame(self, i, rgb, alpha):
        t = i / self.clip.fps
        fr = self.fr
        F, A = fr.place(rgb, alpha)
        pyr = []
        if fr.bg is None:  # full-res source for magnified levels
            Msc = np.float32([[fr.s, 0, -fr.x0 * fr.s + fr.mx], [0, fr.s, -fr.y0 * fr.s + fr.ext]])
            pyr.append((2.0, rgb, alpha.astype(np.float32) / 255, Msc))
        pyr.append((1.0, F, A, np.float32([[1, 0, 0], [0, 1, 0]])))
        f2, a2 = F, A
        for k in (0.5, 0.25, 0.125):
            f2 = cv2.resize(f2, (max(1, f2.shape[1] // 2), max(1, f2.shape[0] // 2)), interpolation=cv2.INTER_AREA)
            a2 = cv2.resize(a2, (f2.shape[1], f2.shape[0]), interpolation=cv2.INTER_AREA)
            pyr.append((k, f2, a2, np.float32([[1 / k, 0, 0], [0, 1 / k, 0]])))

        u, s, r = self.state(t)
        Ps = self.fixed_point(s, r)
        j0 = math.floor(u)
        frac = u - j0
        levels = []
        for j in range(j0 - 1, j0 + DEPTH + 1):
            sig = s ** (j - u)
            if sig < 0.012:
                break
            levels.append((j, sig, (j - u) * r))
        reveal = fx.smooth(t, self.reveal_t, 0.35)
        out = np.zeros((H, W, 3), np.float32)
        m_in = sh_in = M_in = None
        sig_in = 1.0
        for j, sig, th in reversed(levels):
            M = self.affine(Ps, sig, th)
            cs = np.float32([[0, 0], [W, 0], [W, H], [0, H]]) @ M[:, :2].T + M[:, 2]
            bx0, by0 = np.floor(cs.min(0)).astype(int) - 2
            bx1, by1 = np.ceil(cs.max(0)).astype(int) + 2
            bx0, by0, bx1, by1 = max(bx0, 0), max(by0, 0), min(bx1, W), min(by1, H)
            if bx1 - bx0 < 2 or by1 - by0 < 2:
                continue
            bw, bh = bx1 - bx0, by1 - by0
            cands = [e for e in pyr if e[0] >= sig * 0.85]
            k, src, asrc, Ae = cands[-1] if cands else pyr[0]
            Mt = np.vstack([M, [0, 0, 1]]) @ np.vstack([Ae, [0, 0, 1]])
            Mt = Mt[:2].copy()
            Mt[:, 2] -= (bx0, by0)
            base = cv2.warpAffine(src, Mt, (bw, bh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            base = base.astype(np.float32)
            occ = 1.0 if j > j0 else (1 - fx.smooth(frac, 0.25, 0.4) if j == j0 else 0.0)
            sl = (slice(by0, by1), slice(bx0, bx1))
            if sig < 1:  # deeper levels sit a touch darker: reads as depth
                base *= 0.76 + 0.24 * sig ** 0.4
            if m_in is not None:
                keep = base.copy() if occ > 0.01 else None
                base *= (1 - sh_in[sl])[..., None]
                mi = m_in[sl][..., None]
                base = base * (1 - mi) + out[sl] * mi
                # the inner frame's border: dark keyline + bright white edge, thicker than geometry would give
                ol = self.outline @ M_in[:, :2].T + M_in[:, 2] - np.float32([bx0, by0])
                bwid = max(1.6, 10 * sig_in ** 0.55)
                c.polyline(base, ol, (0, 0, 0), bwid + 4, opacity=0.35)
                c.polyline(base, ol, (250, 250, 250), bwid)
                if keep is not None:
                    pa = cv2.warpAffine(asrc, Mt, (bw, bh), flags=cv2.INTER_LINEAR)[..., None] * occ
                    base = keep * pa + base * (1 - pa)
            self.hud(base, j, sig, th, M, bx0, by0, reveal)
            out[sl] = base
            Mm = M.copy()
            Mm[:, 2] -= (bx0, by0)
            m_in = np.zeros((H, W), np.float32)
            m_in[sl] = cv2.warpAffine(self.mask, Mm, (bw, bh), flags=cv2.INTER_LINEAR)
            M_in, sig_in = M, sig
            # soft shadow this frame throws on the one around it (deep and wide, so every level reads as a layer)
            q = 4
            small = cv2.resize(m_in, (W // q, H // q), interpolation=cv2.INTER_AREA)
            sg = max(2.0, 50 * sig / q)
            small = cv2.GaussianBlur(small, (0, 0), sg)
            dy = int(round(30 * sig / q))
            if dy:
                small = np.roll(small, dy, 0)
            sh_in = cv2.resize(small, (W, H), interpolation=cv2.INTER_LINEAR) * 0.85
        img = c.grade(out, contrast=1.07, sat=1.03, warm=0.01)
        # zoom blur on the pushes (and a touch of colour split on the snappy one)
        up, _, _ = self.state(t - 0.6 / self.clip.fps)
        amt = min(0.06, 0.45 * max(0.0, abs(u - up) * math.log(1 / S0) - 0.07))
        if amt > 0.006:
            img = zblur(img, float(Ps[0]), float(Ps[1]), amt)
        if self.expo_k:
            pk = [p for p in self.pushes if p["k"] == self.expo_k[0]][0]
            cs_ = math.exp(-((t - pk["t"] - 0.06) / 0.1) ** 2)
            if cs_ > 0.03:
                img = fx.rgb_shift(img, 7 * cs_, angle=math.degrees(math.atan2(Ps[1] - 960, Ps[0] - 540)))
        img = c.bloom(img, 215, 24, 0.2)
        img = c.vignette(img, 0.3)
        return c.grain(img, i, 2.4)

    def hud(self, base, j, sig, th, M, bx0, by0, reveal):
        """Viewfinder corners and the label of frame j, drawn in the frame's own space."""
        off = np.float32([bx0, by0])

        def tr(pts):
            return np.float32(pts) @ M[:, :2].T + M[:, 2] - off

        if 0.05 < sig < 1.6:
            a_, L = 46, 86
            wdt = max(1.6, 5.0 * sig ** 0.75)
            for (x, y, sx, sy) in ((a_, a_, 1, 1), (W - a_, a_, -1, 1), (W - a_, H - a_, -1, -1), (a_, H - a_, 1, -1)):
                pts = tr([(x, y + sy * L), (x, y), (x + sx * L, y)])
                c.polyline(base, pts, (255, 255, 255), wdt, opacity=0.8 * min(1.0, sig * 3))
        k = self.lab_of.get(j)
        if k is None or reveal <= 0:
            return
        vis = fx.smooth(sig, 0.19, 0.12) * reveal
        if vis <= 0.01:
            return
        big, small, ox, oy = self.tag(k)
        g = sig ** 0.55 if sig < 1 else sig
        anc = tr([(64, 96)])[0]
        e = reveal
        for patch, op, lift in ((big, vis, 0.0), (small, vis * fx.smooth(sig, 0.62, 0.3), 0.0)):
            if op <= 0.01:
                continue
            hh, ww = patch.shape[:2]
            cut = int(hh * (0.2 + 0.8 * e))
            p = patch[:cut] if cut < hh else patch
            hh = p.shape[0]
            a = math.radians(th)
            vx, vy = (ww / 2 - ox) * g, (hh / 2 - oy + (1 - e) * 40) * g
            cx = anc[0] + vx * math.cos(a) - vy * math.sin(a)
            cy = anc[1] + vx * math.sin(a) + vy * math.cos(a)
            fx.put(base, p, cx, cy, g, -th, opacity=op)
