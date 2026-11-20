"""/card-flip-3d: the whole video lives on a thick rounded 3D card floating in a dark studio over a glossy floor.
It flies in from deep space spinning with real motion blur and lands with a spring (the floor flashes); on the
ideas it flips over, once around each axis, to show its back with the idea in huge type, then flips back to the
live video; it floats, dollies and tilts in between, and for the payoff it pushes in under a light sweep while the
last idea rises on the wall behind it."""
import math

import cv2
import numpy as np

import core as c
import fx
from plan import Plan, norm

W, H = c.W, c.H
AMBER = (255, 168, 84)
PERSP = 2200.0
TW, TH = 810, 1440          # card texture
CW, CH = 648, 1152          # card size on screen at rest (z = 0)
THICK = 16
REST = (540.0, 850.0)
FLOOR = 1490


def track_angle(t, events):
    """Sum of spring-eased 180-degree turns."""
    return sum(180 * fx.spring(t, t0, 1.45, 0.5) for t0 in events)


def rise(out, p, x, y, e):
    """Reveal a patch rising out from behind its own bottom edge (mask reveal)."""
    hgt = p.shape[0]
    dy = (1 - e) * hgt * 0.95
    cut = int(hgt - dy)
    if cut > 1:
        c.over(out, p[:cut], x, y + dy)


class Style:
    def __init__(self, clip, opts):
        self.clip = clip
        self.P = P = Plan(clip, opts)
        self.fr = c.Framing(clip, 1215)
        B = self.B = P.beats
        nb = len(B)
        self.T = min(clip.n, len(clip.face)) / clip.fps
        self.cam = fx.Cam(self.fr, [{"t": 0, "zoom": 1.0}], drift=3)
        self.land_t = 0.42 if nb < 1 else min(max(B[0]["key_t"], 0.36), 0.6)
        # flips on the middle beats, alternating axis; each flips over and back
        self.flips = []
        mids = list(range(1, nb - 1)) if nb > 2 else ([1] if nb == 2 else [])
        for n, k in enumerate(mids):
            b = B[k]
            t_in = b["key_t"] - 0.16
            nxt = B[k + 1]["key_t"] if k + 1 < nb else self.T
            t_out = max(t_in + 0.5, min(nxt - 0.44, b["key_t"] + 0.8))
            self.flips.append({"k": k, "axis": "y" if n % 2 == 0 else "x", "in": t_in, "out": t_out,
                               "dir": 1 if n % 4 < 2 else -1})
        self.pay_t = B[-1]["key_t"] if nb > 1 else self.T * 0.7
        self.bg = self._studio()
        self.backs = {}
        self.front_mask = c.rrect(TW, TH, 52)
        self.rim = self._rim()
        self.gloss = (np.clip(1 - np.linspace(0, 1, TH, dtype=np.float32) / 0.45, 0, 1) ** 2 * 0.07)[:, None, None]
        self._tx = {}
        self._cap = {}

    # ---- static studio ----
    def _studio(self):
        ys, xs = fx.grid(H, W)
        bg = fx.gradient(H, W, (20, 19, 22), (7, 7, 9), 90)
        d = np.hypot((xs - 540) / 760, (ys - 760) / 900)
        bg += (np.exp(-d * d * 2.2) * 0.55)[..., None] * np.float32([84, 52, 30])   # warm key behind the card
        d2 = np.hypot((xs - 980) / 500, (ys - 300) / 700)
        bg += (np.exp(-d2 * d2 * 2.0) * 0.35)[..., None] * np.float32([30, 44, 90])  # cool rim from the right
        fl = ys >= FLOOR
        u = np.clip((ys - FLOOR) / (H - FLOOR), 0, 1)
        floor = (np.float32([16, 15, 17]) * (1 - u[..., None] * 0.6) +
                 (np.exp(-((xs - 540) / 520) ** 2 - ((ys - FLOOR - 60) / 170) ** 2) * 0.6)[..., None]
                 * np.float32([60, 42, 28]))
        bg = np.where(fl[..., None], floor, bg)
        hz = np.exp(-((ys - FLOOR) / 2.2) ** 2) * np.exp(-((xs - 540) / 600) ** 2) * 26  # thin horizon glint
        bg += hz[..., None]
        bg = cv2.GaussianBlur(bg, (0, 0), 1.2)
        rng = np.random.default_rng(3)
        return (bg + rng.normal(0, 1.0, bg.shape)).astype(np.float32)

    def _rim(self):
        m = self.front_mask
        inner = np.pad(c.rrect(TW - 6, TH - 6, 49), 3)
        ring = np.clip(m - inner, 0, 1)
        return c.fill(ring, (255, 255, 255), 0.22)

    # ---- type ----
    def tx(self, s, key="grotesk", size=40, color=(255, 255, 255), wght=None, tracking=0.0):
        kk = (s, key, size, color, wght, tracking)
        if kk not in self._tx:
            self._tx[kk] = fx.text(s, key, size, color, wght, tracking)
        return self._tx[kk]

    def back_face(self, k):
        """The back of the card: the idea in huge type."""
        if k in self.backs:
            return self.backs[k]
        b = self.B[k]
        g = fx.gradient(TH, TW, (34, 28, 26), (9, 9, 11), 70)
        ys, xs = fx.grid(TH, TW)
        d = np.hypot((xs - 80) / 700, (ys - TH + 80) / 760)
        g += (np.exp(-d * d * 2.4) * 0.8)[..., None] * np.float32([120, 62, 18])
        sheen = np.exp(-(((xs * 0.8 + ys * 0.6) / (TW * 0.8 + TH * 0.6) - 0.32) / 0.07) ** 2) * 10
        g += sheen[..., None]
        p = np.concatenate([g * self.front_mask[..., None], self.front_mask[..., None] * 255], -1)
        num = self.tx(f"{k + 1:02d}", "serif-i", 170, AMBER)
        c.over(p, num, 56, 40)
        kick = (self.P.kicker or "").upper()
        if kick:
            kp = self.tx(kick, "mono", 38, (200, 196, 190), 520, 0.14)
            if kp.shape[1] > TW - 120:
                kp = self.tx(kick, "mono", int(38 * (TW - 120) / kp.shape[1]), (200, 196, 190), 520, 0.14)
            c.over(p, kp, 60, TH - 120)
        words = b["label"].split()
        lines = [" ".join(words)]
        if len(words) > 1 and c.text_w(lines[0], "grotesk", 150, 800) > TW - 120:
            best = min(range(1, len(words)), key=lambda j: max(len(" ".join(words[:j])), len(" ".join(words[j:]))))
            lines = [" ".join(words[:best]), " ".join(words[best:])]
        size = min(220, min(c.fit_size(ln, "grotesk", TW - 120, 800, -0.02) for ln in lines))
        lh = int(size * 1.0)
        y0 = TH * 0.5 - lh * len(lines) / 2 - size * 0.12
        for n, ln in enumerate(lines):
            c.over(p, self.tx(ln, "grotesk", size, (255, 255, 255), 800, -0.02), 54, y0 + n * lh)
        yy = int(y0 + len(lines) * lh + 46)
        c.over(p, c.fill(np.ones((5, 120), np.float32), AMBER), 60, yy)
        sub = b.get("sub", "")
        if sub:
            c.over(p, self.tx(sub, "mono", 38, (235, 230, 222), 560, 0.16), 60, yy + 30)
        ring = np.clip(self.front_mask - np.pad(c.rrect(TW - 6, TH - 6, 49), 3), 0, 1)
        c.over(p, c.fill(ring, AMBER, 0.35))
        self.backs[k] = p
        return p

    def caption(self, b):
        key = b["text"]
        if key in self._cap:
            return self._cap[key]
        ws = [w["w"].strip(" ,.") for w in b["words"]] or b["text"].split()
        size = 56
        while True:
            ps = [self.tx(w, "grotesk", size, AMBER if norm(w) in self.P.hot else (245, 242, 238), 700) for w in ws]
            gap = int(size * 0.26)
            wd = sum(p.shape[1] for p in ps) + gap * (len(ps) - 1)
            if wd <= 960 or size <= 34:
                break
            size -= 4
        out = c.layer(max(p.shape[0] for p in ps), wd)
        x = 0
        for p in ps:
            c.over(out, p, x, 0)
            x += p.shape[1] + gap
        self._cap[key] = out
        return out

    # ---- card pose ----
    def pose(self, t):
        """Returns dict: X, Y, Z (world offsets), rx, ry, rz, back (which face texture), axis."""
        ez = fx.spring(t, -0.12, 1.35, 0.6)
        es = fx.expo(t, -0.12, 0.85)
        X = c.lerp(-340, 0, ez)
        Y = c.lerp(420, 0, ez)
        Z = c.lerp(2600, 0, ez)
        ry = -380 * (1 - es)
        rz = -28 * (1 - ez)
        rx = 18 * (1 - ez)
        # idle float + slow dolly / tilt
        rx += 6 * math.sin(t * 0.8 + 0.3)
        ry += 10 * math.sin(t * 0.55 + 0.9)
        rz += 0.8 * math.sin(t * 0.7)
        Y += 9 * math.sin(t * 1.15)
        Z += -60 * fx.smooth(t, 0.8, 2.6)
        axis, back_k = "y", 0
        for f in self.flips:
            if t < f["in"] - 0.02:
                continue
            a = track_angle(t, (f["in"], f["out"])) * f["dir"]
            if f["axis"] == "y":
                ry += a
            else:
                rx += a
            lift = math.sin(math.pi * c.prog(t, f["in"], 0.55)) + math.sin(math.pi * c.prog(t, f["out"], 0.55))
            Z -= 230 * lift
            rz += (3 if f["axis"] == "y" else -2) * lift
            axis, back_k = f["axis"], f["k"]
        q = fx.smooth(t, self.pay_t - 0.1, 0.9)
        Z -= 150 * q
        Y += 92 * q
        rx += 6 * q
        return {"X": X, "Y": Y, "Z": Z, "rx": rx, "ry": ry, "rz": rz, "axis": axis, "back": back_k}

    # ---- card drawing ----
    def corners(self, p):
        R = fx.rot3(p["rx"], p["ry"], p["rz"])
        hw, hh, ht = CW / 2, CH / 2, THICK / 2
        loc = np.float32([[-hw, -hh, -ht], [hw, -hh, -ht], [hw, hh, -ht], [-hw, hh, -ht],
                          [-hw, -hh, ht], [hw, -hh, ht], [hw, hh, ht], [-hw, hh, ht]])
        P = loc @ R.T + np.float32([REST[0] - 540 + p["X"], REST[1] - 960 + p["Y"], p["Z"]])
        zz = P[:, 2] + PERSP
        scr = P[:, :2] * (PERSP / np.maximum(zz, 50))[:, None] + np.float32([540, 960])
        nrm = R @ np.float32([0, 0, -1])
        ctr = P.mean(0)
        view = ctr - np.float32([0, 0, -PERSP])
        facing = -float(np.dot(nrm, view) / (np.linalg.norm(view) + 1e-6))
        return scr, P, R, facing

    def draw_card(self, L, tex_front, p):
        scr, P3, R, facing = self.corners(p)
        if (P3[:, 2] + PERSP < 60).any():
            return
        cam = np.float32([0, 0, -PERSP])
        # edges (the card's thickness), drawn first
        sides = [((0, 1), (0, -1, 0)), ((1, 2), (1, 0, 0)), ((2, 3), (0, 1, 0)), ((3, 0), (-1, 0, 0))]
        for (i, j), n in sides:
            nw = R @ np.float32(n)
            ctr = (P3[i] + P3[j] + P3[i + 4] + P3[j + 4]) / 4
            if np.dot(nw, ctr - cam) >= 0:
                continue
            lit = 0.5 + 0.5 * max(0.0, float(-nw[1]) * 0.6 + float(-nw[0]) * 0.4)
            col = np.float32([46, 42, 40]) * (0.7 + 0.8 * lit)
            poly = np.stack([scr[i], scr[j], scr[j + 4], scr[i + 4]])
            self.fill_poly(L, poly, col)
        if facing >= 0:
            tex, quad = tex_front, scr[:4]
        else:
            bk = self.back_face(p["back"])
            tex = bk[:, ::-1] if p["axis"] == "y" else bk[::-1, :]
            quad = scr[4:]
        shade = 0.55 + 0.45 * abs(facing) ** 0.6
        self.warp(L, tex, quad, shade)

    @staticmethod
    def fill_poly(L, poly, col):
        x0, y0 = np.floor(poly.min(0)).astype(int) - 2
        x1, y1 = np.ceil(poly.max(0)).astype(int) + 2
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if x1 - x0 < 1 or y1 - y0 < 1:
            return
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillConvexPoly(m, ((poly - [x0, y0]) * 16).astype(np.int32), 255, cv2.LINE_AA, 4)
        c.over(L, c.fill(m.astype(np.float32) / 255, col), x0, y0)

    @staticmethod
    def warp(L, tex, quad, shade):
        x0, y0 = np.floor(quad.min(0)).astype(int) - 2
        x1, y1 = np.ceil(quad.max(0)).astype(int) + 2
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if x1 - x0 < 2 or y1 - y0 < 2:
            return
        h, w = tex.shape[:2]
        M = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]),
                                        (quad - np.float32([x0, y0])).astype(np.float32))
        out = cv2.warpPerspective(tex, M, (int(x1 - x0), int(y1 - y0)), flags=cv2.INTER_LINEAR, borderValue=0)
        if shade < 0.999:
            out[..., :3] *= shade
        c.over(L, out, x0, y0)

    # ---- frame ----
    def frame(self, i, rgb, alpha):
        t = i / self.clip.fps
        fps = self.clip.fps
        img, _ = self.cam.place(i, t, rgb, alpha)
        img = c.grade(img, contrast=1.08, sat=1.03, warm=0.01)
        tex = cv2.resize(img, (TW, TH), interpolation=cv2.INTER_AREA) * (1 + self.gloss)
        tex = np.concatenate([tex * self.front_mask[..., None], self.front_mask[..., None] * 255], -1)
        c.over(tex, self.rim)
        sw = c.prog(t, self.pay_t + 0.12, 0.85)  # the one light sweep, on the payoff
        if 0 < sw < 1:
            tex = fx.light_sweep(tex, sw, width=0.065, angle=28, strength=0.38)
        out = self.bg.copy()
        # payoff: the last idea rises on the wall behind the card
        self.wall_title(out, t)
        # card, temporally supersampled when it moves fast (real motion blur)
        dt = 0.6 / fps
        p0, p1 = self.pose(t), self.pose(t - dt)
        s0, s1 = self.corners(p0)[0], self.corners(p1)[0]
        disp = float(np.abs(s0 - s1).max())
        n = int(c.clamp(1 + disp / 12, 1, 8))
        L = c.layer()
        if n == 1:
            self.draw_card(L, tex, p0)
        else:
            for k in range(n):
                Lk = c.layer()
                self.draw_card(Lk, tex, self.pose(t - dt * k / (n - 1)))
                L += Lk
            L /= n
        # contact shadow + floor reflection
        bottom = float(s0[:4, 1].max())
        gap = max(0.0, FLOOR - bottom)
        wpx = float(np.ptp(s0[:4, 0]))
        sa = 0.75 * math.exp(-gap / 260) * c.clamp(wpx / 300)
        if sa > 0.01:
            sh_w, sh_h = int(wpx * 0.95) + 120, 150
            ys, xs = fx.grid(sh_h, sh_w)
            m = np.exp(-(((xs - sh_w / 2) / (wpx * 0.42 + 30)) ** 2 + ((ys - sh_h / 2) / (16 + gap * 0.08)) ** 2) * 1.6)
            cxs = float(s0[:4, 0].mean())
            c.over(out, c.fill(m, (0, 0, 0), sa), cxs - sh_w / 2, FLOOR - sh_h / 2 + 6)
        self.reflect(out, L)
        self.floor_ring(out, t)
        c.over(out, L)
        # caption on the floor
        self.floor_caption(out, t)
        out = c.bloom(out, 210, 26, 0.25)
        out = c.vignette(out, 0.35)
        return c.grain(out, i, 2.6)

    def reflect(self, out, L):
        n = H - FLOOR
        src = L[max(0, FLOOR - n):FLOOR][::-1]
        n = src.shape[0]
        if n < 2 or src[..., 3].max() < 1:
            return
        fade = (np.linspace(1, 0, n, dtype=np.float32) ** 1.7 * 0.3)[:, None, None]
        r = cv2.GaussianBlur(src * fade, (0, 0), 2.5)
        c.over(out, r, 0, FLOOR)

    def floor_ring(self, out, t):
        p = c.prog(t, self.land_t - 0.06, 0.8)
        if not 0 < p < 1:
            return
        e = c.ease_out_expo(p)
        rx, ry = 120 + 640 * e, 18 + 90 * e
        hw, hh = int(rx + 40), int(ry + 40)
        ys, xs = fx.grid(2 * hh, 2 * hw)
        d = np.hypot((xs - hw) / rx, (ys - hh) / ry)
        ring = np.exp(-((d - 1) / 0.06) ** 2) * (1 - p) ** 1.5 + np.exp(-d * d * 3) * 0.5 * (1 - p) ** 2
        fx.add(out, c.fill(ring, AMBER), 540 - hw, FLOOR + 4 - hh, 0.9)

    def wall_title(self, out, t):
        q = fx.spring(t, self.pay_t - 0.02, 1.6, 0.7)
        if q <= 0.001:
            return
        b = self.B[-1]
        lab = b["label"]
        size = min(168, c.fit_size(lab, "grotesk", 960, 800, -0.03))
        p = self.tx(lab, "grotesk", size, (255, 255, 255), 800, -0.03)
        y = 236 - p.shape[0] * 0.8
        rise(out, p, 540 - p.shape[1] / 2, y, c.clamp(q))
        sub = b.get("sub", "")
        if sub:
            sp = self.tx(sub, "mono", 30, AMBER, 600, 0.22)
            c.over(out, sp, 540 - sp.shape[1] / 2, 34, fx.smooth(t, self.pay_t + 0.12, 0.3))

    def floor_caption(self, out, t):
        k, b, _ = self.P.at(t)
        cap = self.caption(b)
        e = fx.expo(t, b["t0"] + 0.08 if k else 0.25, 0.45)
        y = 1712
        if k == len(self.P.beats) - 1 and len(self.P.beats) > 1:  # the payoff title takes over on the wall
            e *= 1 - fx.smooth(t, self.pay_t + 0.05, 0.25)
        rise(out, cap, 540 - cap.shape[1] / 2, y - cap.shape[0] / 2, e)
        if k:
            pc = self.caption(self.P.beats[k - 1])
            o = 1 - c.clamp((t - b["t0"]) / 0.09)
            if o > 0:
                c.over(out, pc, 540 - pc.shape[1] / 2, y - pc.shape[0] / 2 - (1 - o) * 40, o)
