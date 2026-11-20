"""/ranking-format: premium countdown. A giant chrome 3D-extruded rank numeral slams through the camera, then each new rank swings in on a different hinge with real depth and motion blur; every beat's label flies down into a frosted liquid-glass tier board that fills from the bottom up; #1 turns gold with a light sweep and a physics confetti burst."""
import math

import cv2
import numpy as np

import core as c
import fx
from plan import Plan

PERSP = 1700.0
WHITE = (255, 255, 255)
GOLD = (240, 192, 88)
CHROME = [(0.0, (255, 255, 255)), (0.36, (214, 218, 228)), (0.5, (120, 124, 138)), (0.56, (172, 176, 188)),
          (0.85, (236, 238, 244)), (1.0, (196, 200, 210))]
GILT = [(0.0, (255, 246, 205)), (0.36, (246, 206, 112)), (0.5, (150, 98, 26)), (0.56, (206, 150, 56)),
        (0.85, (255, 222, 140)), (1.0, (214, 160, 66))]
NUM_C = (218, 300)


def stops(u, st):
    out = np.zeros(u.shape + (3,), np.float32)
    for (u0, c0), (u1, c1) in zip(st[:-1], st[1:]):
        m = (u >= u0) & (u <= u1)
        x = ((u - u0) / max(u1 - u0, 1e-6))[..., None]
        out[m] = (np.float32(c0) * (1 - x) + np.float32(c1) * x)[m]
    return out


def proj3(pts, P):
    R = fx.rot3(P["rx"], P["ry"], P["rz"])
    piv = np.float32([P["px"], P["py"], 0])
    Q = (pts - piv) @ R.T + piv
    Q = Q * P["s"]
    zz = Q[:, 2] + PERSP + P["z"]
    return Q[:, :2] * (PERSP / np.maximum(zz, 1))[:, None] + np.float32([P["cx"], P["cy"]]), zz


class Style:
    def __init__(self, clip, opts):
        self.clip = clip
        self.P = P = Plan(clip, opts)
        self.fr = fr = c.Framing(clip, 1215)
        self.B = B = P.beats
        self.n = n = len(B)
        self.N = min(clip.n, len(clip.face))
        self.dur = self.N / clip.fps
        self.slam_t = 0.34
        keys = [{"t": 0, "zoom": 1.0}, {"t": self.slam_t - 0.02, "zoom": 1.07, "ease": "spring", "freq": 2.4, "damp": 0.45},
                {"t": 0.8, "zoom": 1.03, "ease": "smooth", "dur": 0.9}]
        for k in range(1, n):
            L = B[k]["key_t"]
            last = k == n - 1
            keys += [{"t": L - 0.05, "zoom": 1.11 if last else 1.06 + 0.01 * k, "dy": -10 if last else 0,
                      "ease": "spring", "freq": 1.9, "damp": 0.55},
                     {"t": L + 0.45, "zoom": 1.08 if last else 1.04 + 0.01 * k, "ease": "smooth", "dur": 0.9}]
        self.cam = fx.Cam(fr, keys, drift=4)
        self.nums = {}
        for k in range(n):
            r = n - k
            self.nums[r] = self._numeral(r, gold=(r == 1))
        self._tx = {}
        # the glass tier board: sized by the number of beats, rows >= 96 px
        self.row_h = rh = int(min(138, max(96, 600 / n)))
        self.HEAD = 92
        self.BX, self.BW = 30, 1020
        self.BH = self.HEAD + n * rh + 22
        self.BY = c.H - 34 - self.BH
        self.cs = int(min(108, rh - 28))
        self.LX = self.BX + 30 + self.cs + 32
        self.stack = rh >= 118
        room = self.BX + self.BW - 36 - self.LX
        self.labs = []
        for k, b in enumerate(B):
            lab = b["label"]
            si = int(min(96, c.fit_size(lab, "grotesk", 1080 - 456 - 40, 860, -0.02)))
            sub = (b.get("sub") or "").upper()
            if self.stack:
                sr = int(max(44, min(66, c.fit_size(lab, "grotesk", room, 800, -0.02))))
            else:
                sub_w = c.text_w(sub, "mono", 26, 560, 0.14) + 30 if sub else 0
                sr = int(min(rh * 0.5, c.fit_size(lab, "grotesk", room - sub_w, 800, -0.02)))
                if sr < 44:
                    sub, sr = "", int(max(44, min(rh * 0.5, c.fit_size(lab, "grotesk", room, 800, -0.02))))
            self.labs.append({"si": si, "sr": sr, "sub": sub})
        self.fly = []
        for k, b in enumerate(B):
            nxt = B[k + 1]["key_t"] if k + 1 < n else self.dur + 1
            self.fly.append(max(b["key_t"] + 0.25, min(b["key_t"] + 0.6, nxt - 0.42)))
        if n > 1:  # the payoff label stays up as the end card; its row fills in on its own
            self.fly[-1] = B[-1]["key_t"] + 0.12
        self.land = [f + 0.36 for f in self.fly]
        rng = np.random.default_rng(21)
        self.conf = []
        for _ in range(120):
            ang = rng.uniform(-math.pi * 0.95, -math.pi * 0.05) if rng.random() < 0.8 else rng.uniform(0, math.pi)
            sp = rng.uniform(700, 2100)
            self.conf.append(dict(vx=math.cos(ang) * sp, vy=math.sin(ang) * sp - 300, w=rng.uniform(12, 20),
                                  h=rng.uniform(22, 36), spin=rng.uniform(6, 16), ph=rng.uniform(0, 6.28),
                                  rot=rng.uniform(0, 360), vr=rng.uniform(-420, 420),
                                  col=[(246, 206, 112), (255, 238, 196), (255, 255, 255), (200, 140, 48),
                                       (255, 222, 140)][int(rng.integers(0, 5))], sway=rng.uniform(20, 70)))

    # ---------- assets ----------
    def tx(self, s, key, size, color=WHITE, wght=None, tracking=0.0):
        kk = (s, key, size, color, wght, tracking)
        if kk not in self._tx:
            self._tx[kk] = c.text_patch(s, key, size, color, wght, tracking)
        return self._tx[kk]

    def _numeral(self, r, gold):
        size = 400
        d, _ = c.text_patch(str(r), "grotesk", size, WHITE, 900, -0.04, 10)
        hs, _ = c.text_patch("#", "grotesk", int(size * 0.4), WHITE, 800, 0.0, 10)

        def ink(p):
            ys = np.nonzero(p[..., 3].max(1) > 10)[0]
            xs = np.nonzero(p[..., 3].max(0) > 10)[0]
            return p[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1, 3] / 255

        a_d, a_h = ink(d), ink(hs)
        gap = 14
        pad = 16
        h = a_d.shape[0] + 2 * pad
        w = a_h.shape[1] + gap + a_d.shape[1] + 2 * pad
        A = np.zeros((h, w), np.float32)
        A[pad:pad + a_h.shape[0], pad:pad + a_h.shape[1]] = a_h
        A[pad:pad + a_d.shape[0], pad + a_h.shape[1] + gap:pad + a_h.shape[1] + gap + a_d.shape[1]] = \
            np.maximum(A[pad:pad + a_d.shape[0], pad + a_h.shape[1] + gap:pad + a_h.shape[1] + gap + a_d.shape[1]], a_d)
        u = np.linspace(0, 1, h, dtype=np.float32)[:, None] * np.ones((1, w), np.float32)
        col = stops(u, GILT if gold else CHROME)
        # bevel: lit from the top-left
        hgt = cv2.GaussianBlur(np.clip(cv2.distanceTransform((A > 0.5).astype(np.uint8), cv2.DIST_L2, 5) / 9, 0, 1),
                               (0, 0), 1.2)
        gy, gx = np.gradient(hgt)
        nrm = np.hypot(gx, gy) + 1e-6
        lit = -(gx * -0.6 + gy * -0.8) / nrm * np.clip(nrm * 9, 0, 1)
        col = col + np.clip(lit, 0, 1)[..., None] * 70 - np.clip(-lit, 0, 1)[..., None] * 70
        ys, xs = fx.grid(h, w)
        band = np.exp(-(((xs * 0.55 + ys) / (w * 0.55 + h)) - 0.3) ** 2 / 0.004) * 60
        col = np.clip(col + band[..., None], 0, 255)
        face = np.concatenate([col * A[..., None], A[..., None] * 255], -1)
        side = ((150, 98, 26), (70, 40, 8)) if gold else ((92, 96, 108), (24, 25, 30))
        return {"face": face, "sil": A, "side": side, "w": w, "h": h}

    # ---------- numeral motion ----------
    def pose(self, k, t):
        n = self.n
        r = n - k
        N = self.nums[r]
        w, h = N["w"], N["h"]
        P = dict(cx=NUM_C[0], cy=NUM_C[1], rx=0.0, ry=0.0, rz=0.0, s=1.0, z=0.0, px=0.0, py=0.0, op=1.0)
        L = self.B[k]["key_t"]
        if k == 0:  # slams through the camera
            e = fx.expo(t, 0.0, 0.36) if t < self.slam_t else 1.0
            if t < self.slam_t:
                x = c.clamp(t / self.slam_t)
                e = 1 - (1 - x) ** 2.6
            P["z"] = -1460 * (1 - e)
            P["rz"] = -14 * (1 - e)
            P["cx"] += 300 * (1 - e)
            P["cy"] += 360 * (1 - e)
        elif k == n - 1:  # rises with a spin, settles on a spring
            ts = L - 0.2
            if t < ts:
                return None
            e = fx.spring(t, ts, 1.6, 0.55)
            P["cy"] -= 640 * (1 - e)
            P["ry"] = -62 * (1 - fx.spring(t, ts, 1.3, 0.62))
            P["rx"] = -30 * (1 - e)
            P["s"] = 0.8 + 0.2 * e
        else:  # swings in on a hinge: top, then side, alternating
            ts = L - 0.2
            if t < ts:
                return None
            e = fx.spring(t, ts, 1.5, 0.42)
            if (k - 1) % 2 == 0:
                P["py"] = -h / 2
                P["rx"] = -105 * (1 - e)
            else:
                P["px"] = -w / 2
                P["ry"] = 110 * (1 - e)
        # idle float so the extrusion keeps catching perspective
        P["ry"] += 7 * math.sin(t * 1.7 + k)
        P["rx"] += 3.5 * math.sin(t * 1.2 + 0.6 * k)
        P["cy"] += 6 * math.sin(t * 1.4 + k)
        # exit when the next rank arrives: flips away on its top hinge
        if k + 1 < n:
            te = self.B[k + 1]["key_t"] - 0.24
            if t >= te:
                q = fx.smooth(t, te, 0.2)
                q = q * q
                P["py"] = -h / 2
                P["px"] = 0.0
                P["rx"] += 92 * q
                P["op"] = 1 - c.clamp((q - 0.8) / 0.2)
                if q >= 0.99:
                    return None
        return P

    def draw_numeral(self, img, k, t, sweep=0.0):
        P0 = self.pose(k, t)
        if P0 is None:
            return
        P1 = self.pose(k, t - 0.55 / self.clip.fps) or P0
        r = self.n - k
        N = self.nums[r]
        w, h = N["w"], N["h"]
        K, D = 16, 74
        base = np.float32([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0], [w / 2, h / 2, 0], [-w / 2, h / 2, 0]])
        quads = []
        for j in range(K + 1):
            pts = base.copy()
            pts[:, 2] = D * (1 - j / K)
            q, zz = proj3(pts, P0)
            if (zz < 90).any():
                return
            quads.append(q)
        q1, _ = proj3(base, P1)
        dq = quads[-1] - q1
        jj = int(np.argmax(np.hypot(dq[:, 0], dq[:, 1])))
        mv = dq[jj] * 0.85
        allq = np.concatenate(quads)
        m = int(abs(mv).max()) + 40
        x0, y0 = np.floor(allq.min(0)).astype(int) - m
        x1, y1 = np.ceil(allq.max(0)).astype(int) + m
        x0, y0, x1, y1 = max(x0, -60), max(y0, -60), min(x1, c.W + 60), min(y1, c.H + 60)
        if x1 - x0 < 4 or y1 - y0 < 4:
            return
        bw, bh = int(x1 - x0), int(y1 - y0)
        tmp = np.zeros((bh, bw, 4), np.float32)
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        sil = N["sil"]
        lo, hi = np.float32(N["side"][1]), np.float32(N["side"][0])
        for j in range(K):
            Mh = cv2.getPerspectiveTransform(src, (quads[j] - np.float32([x0, y0])).astype(np.float32))
            a = cv2.warpPerspective(sil, Mh, (bw, bh), flags=cv2.INTER_LINEAR, borderValue=0)[..., None]
            col = lo + (hi - lo) * (j / K) ** 1.4
            tmp[..., :3] = tmp[..., :3] * (1 - a) + col * a
            tmp[..., 3:] = tmp[..., 3:] * (1 - a) + 255 * a
        face = N["face"]
        if sweep > 0:
            face = fx.light_sweep(face, sweep, width=0.12, angle=24, strength=0.9)
        Mh = cv2.getPerspectiveTransform(src, (quads[-1] - np.float32([x0, y0])).astype(np.float32))
        c.over(tmp, cv2.warpPerspective(face, Mh, (bw, bh), flags=cv2.INTER_LINEAR, borderValue=0))
        # motion blur: dominant corner direction; zoom blur while slamming
        s0 = PERSP / (PERSP + P0["z"])
        s1 = PERSP / (PERSP + P1["z"])
        if abs(s0 - s1) / s0 > 0.02:
            ctr = quads[-1].mean(0) - np.float32([x0, y0])
            tmp = fx.zoom_blur(tmp, float(ctr[0]), float(ctr[1]), min(0.5, (s0 - s1) / s0 * 2.2), n=17)
        elif math.hypot(*mv) > 2:
            tmp = fx.dir_blur(tmp, float(mv[0]), float(mv[1]))
        sh, (ox, oy) = c.shadow(tmp, 30, 0.55, 30)
        c.over(img, sh, x0 + ox + 16, y0 + oy, P0["op"])
        c.over(img, tmp, x0, y0, P0["op"])

    # ---------- board ----------
    def board_y(self, t):
        e = fx.spring(t, 0.16, 1.35, 0.72)
        return self.BY + (1 - e) * 900

    def row_y(self, k):
        r = self.n - k  # rank -> row (1 at top)
        return self.BY + self.HEAD + (r - 1) * self.row_h + self.row_h / 2

    def shadowed(self, img, p, x, y, op=1.0, scale=1.0):
        """Text with a soft dark shadow so it reads on glass; (x, y) is the patch's left-centre."""
        w = p.shape[1] * scale
        fx.put(img, p, x + w / 2, y, scale=scale, opacity=op, shadow=0.55, sblur=6, sdy=3)

    def board(self, img, t, k):
        by = self.board_y(t)
        if by >= c.H:
            return
        BX, BW, BH, rh, cs = self.BX, self.BW, self.BH, self.row_h, self.cs
        dy = by - self.BY
        y0 = int(by)
        msk = c.rrect(BW, BH, 52)
        fx.glass_shadow(img, msk, BX, y0, blur=40, a=0.5, dy=26)
        g = fx.glass(img, msk, BX, y0, bevel=58, refract=1.4, magnify=0.035, frost=8, chroma=0.6,
                     tint=(226, 232, 244), tint_a=0.17, spec=1.15, rim=1.0, depth=0.08)
        c.over(img, g, BX, y0)
        # top gloss and a crisp bright rim
        u = np.clip(1 - np.arange(BH, dtype=np.float32) / (BH * 0.42), 0, 1) ** 2 * 0.09
        gl = c.fill(msk * u[:, None], WHITE)
        fx.add(img, gl, BX, y0, 1.0)
        ring = np.clip(msk - np.pad(c.rrect(BW - 5, BH - 5, 50), ((2, 3), (2, 3))), 0, 1)
        c.over(img, c.fill(ring, WHITE, 0.55), BX, y0)
        n = self.n
        hd = f"TOP {n}" + (f"  ·  {self.P.kicker.upper()}" if self.P.kicker else "")
        pp, base = self.tx(hd, "mono", 30, WHITE, 620, 0.2)
        c.over(img, pp, BX + 34, by + 58 - base, 0.92)
        lp = c.fill(np.ones((2, BW - 68), np.float32), WHITE, 0.28)
        c.over(img, lp, BX + 34, by + self.HEAD - 6)
        for j in range(1, n):
            ly = self.BY + self.HEAD + j * rh + dy
            c.over(img, c.fill(np.ones((1, BW - 68), np.float32), WHITE, 0.12), BX + 34, ly)
        # the highlight bar slides to the newest row
        landed = [j for j in range(n) if t >= self.land[j]]
        if landed:
            cur = None
            for j in landed:
                e = fx.spring(t, self.land[j], 2.0, 0.65)
                yj = self.row_y(j) + dy
                cur = yj if cur is None else cur + (yj - cur) * e
            bar = c.fill(c.rrect(BW - 36, int(rh - 14), 28), WHITE, 0.12)
            c.over(img, bar, BX + 18, cur - (rh - 14) / 2)
        for j in range(n):
            r = n - j
            y = self.row_y(j) + dy
            land = self.land[j]
            done = t >= land
            gold = r == 1
            cx = BX + 30 + cs / 2
            # landing glow
            if done:
                d = t - land
                ga = (1 - math.exp(-d / 0.03)) * math.exp(-d / 0.26) * 0.75
                if ga > 0.01:
                    gw, gh = BW - 36, int(rh - 10)
                    pad = 30
                    m = cv2.GaussianBlur(np.pad(c.rrect(gw, gh, 28), pad), (0, 0), 16)
                    col = GOLD if gold else (255, 255, 255)
                    fx.add(img, c.fill(m, col), BX + 18 - pad, y - gh / 2 - pad, ga)
            # rank chip
            e = fx.spring(t, land, 2.4, 0.42) if done else 0.0
            num_col = (30, 20, 6) if gold else WHITE
            if e < 1:
                empty = c.card(cs, cs, 26, WHITE, 0.06, stroke=(255, 255, 255, 0.42), stroke_w=3)
                nm, nb = self.tx(str(r), "grotesk", int(cs * 0.5), WHITE, 760, -0.02)
                c.over(empty, nm, (cs - nm.shape[1]) / 2 + 1, cs / 2 - nb + cs * 0.18, 0.45)
                c.over(img, empty, BX + 30, y - cs / 2, 1 - min(1, e))
            if e > 0:
                badge = c.card(cs, cs, 26, GOLD if gold else WHITE, 1.0 if gold else 0.22,
                               stroke=(255, 255, 255, 0.7), stroke_w=2)
                nm, nb = self.tx(str(r), "grotesk", int(cs * 0.56), num_col, 840, -0.02)
                c.over(badge, nm, (cs - nm.shape[1]) / 2 + 1, cs / 2 - nb + cs * 0.2)
                fx.put(img, badge, cx, y, scale=0.5 + 0.5 * e, opacity=min(1, e * 2), shadow=0.35, sblur=10, sdy=5)
            lab = self.labs[j]
            sr = lab["sr"]
            sub = lab["sub"]
            if not done:
                ph = c.fill(c.rrect(int(min(420, self.BX + BW - 40 - self.LX)), 22, 11), WHITE, 0.1)
                c.over(img, ph, self.LX, y - 11)
                continue
            x = c.clamp((t - land) / 0.3)
            bump = 1 + 0.13 * math.sin(math.pi * x)
            p, base = self.tx(self.B[j]["label"], "grotesk", sr, WHITE, 800, -0.02)
            if gold:
                sw = c.prog(t, land + 0.12, 0.8)
                if 0 < sw < 1:
                    p = fx.light_sweep(p, sw, 0.14, 24, 1.0)
            if self.stack and sub:
                ly = y - 22  # label above, sub beneath
            else:
                ly = y
            pcy = ly + sr * 0.36 - base + p.shape[0] / 2
            self.shadowed(img, p, self.LX, pcy, 1.0, bump)
            if sub:
                q = fx.expo(t, land + 0.1, 0.45)
                sp, sb = self.tx(sub, "mono", 30 if self.stack else 26, GOLD if gold else (240, 242, 248), 620, 0.14)
                if self.stack:
                    sx, sy = self.LX + 2 + (1 - q) * 24, ly + sr * 0.36 + 50
                else:
                    sx, sy = self.BX + BW - 36 - sp.shape[1] + (1 - q) * 24, y + 10
                c.over(img, sp, sx, sy - sb, q)

    def labels(self, img, t, k):
        """The beat label pops in beside the numeral, then flies down into its row."""
        for j in range(self.n):
            b = self.B[j]
            L = b["key_t"]
            tf = self.fly[j]
            final = j == self.n - 1 and self.n > 1
            if t < L - 0.06 or (t >= self.land[j] and not final):
                continue
            lab = self.labs[j]
            p, base = self.tx(b["label"], "grotesk", lab["si"], WHITE, 860, -0.02)
            stacked = self.stack and lab["sub"]

            def at(tt):
                ein = fx.expo(tt, L - 0.06, 0.4)
                ef = 0.0 if final else fx.expo(tt, tf, 0.36)
                x0 = 456 + p.shape[1] / 2 + (1 - ein) * 80
                y0 = 300 - base + p.shape[0] / 2
                sr = lab["sr"] / lab["si"]
                x1 = self.LX + p.shape[1] * sr / 2
                yrow = self.row_y(j) - (22 if stacked else 0)
                y1 = yrow + lab["sr"] * 0.36 - base * sr + p.shape[0] * sr / 2
                arc = math.sin(math.pi * c.clamp(ef)) * 330  # swings out around the face
                return (x0 + (x1 - x0) * ef + arc, y0 + (y1 - y0) * ef, 1 + (sr - 1) * ef, ein)

            x, y, s, ein = at(t)
            xp, yp, _, _ = at(t - 0.022)
            op = min(1.0, ein * 1.4)
            fx.put(img, p, x, y, scale=s, opacity=op, mv=((x - xp) * 0.9, (y - yp) * 0.9), shadow=0.4, sblur=14, sdy=6)
            sub = lab["sub"] or ""
            if sub and (t < tf + 0.05 or final):
                q = fx.expo(t, L, 0.4) * (1.0 if final else 1 - fx.smooth(t, tf - 0.05, 0.12))
                pp, pb = self.tx(sub, "mono", 30, GOLD, 560, 0.24)
                c.over(img, pp, 458, 300 - lab["si"] * 0.82 - pb - 8 + (1 - q) * 16, q)

    def confetti(self, img, t):
        tb = self.B[-1]["key_t"] + 0.12
        if t < tb or self.n < 2:
            return
        dt = t - tb
        kd, g = 2.6, 2600.0
        ox, oy = NUM_C
        e = math.exp(-kd * dt)
        for p in self.conf:
            x = ox + p["vx"] * (1 - e) / kd + p["sway"] * math.sin(dt * 3 + p["ph"]) * min(1, dt * 2)
            y = oy + g / kd * dt + (p["vy"] - g / kd) * (1 - e) / kd
            if not (-60 < x < c.W + 60 and -60 < y < c.H + 60):
                continue
            vx = p["vx"] * e
            vy = g / kd + (p["vy"] - g / kd) * e
            fl = math.cos(p["ph"] + p["spin"] * dt)
            w = max(2, int(p["w"] * abs(fl)))
            sh = 1.0 if fl > 0 else 0.72
            col = tuple(v * sh for v in p["col"])
            patch = c.fill(np.ones((int(p["h"]), w), np.float32), col)
            fx.put(img, patch, x, y, rot=p["rot"] + p["vr"] * dt, mv=(vx * 0.014, vy * 0.014))

    # ---------- frame ----------
    def frame(self, i, rgb, alpha):
        t = i / self.clip.fps
        img, a = self.cam.place(i, t, rgb, alpha)
        img = c.grade(img, contrast=1.14, sat=0.86, gain=0.84, warm=0.006)
        # floor gradient so the board reads
        g = np.clip((np.arange(c.H, dtype=np.float32) - 1050) / 700, 0, 1) ** 1.5 * 0.55
        img *= (1 - g)[:, None, None]
        k = 0
        for j, b in enumerate(self.B):
            if t >= b["key_t"] - 0.24:
                k = j
        self.board(img, t, k)
        last = self.n - 1
        for j in (k - 1, k):
            if j >= 0:
                sweep = c.prog(t, self.B[last]["key_t"] + 0.3, 0.75) if j == last else 0.0
                self.draw_numeral(img, j, t, sweep)
        self.labels(img, t, k)
        self.confetti(img, t)
        # impact: exposure flash and a decaying shake
        fl = math.exp(-((t - self.slam_t) / 0.05) ** 2) * 0.22 if t >= self.slam_t - 0.04 else 0.0
        if fl > 0.01:
            img = img + (255 - img) * fl
        img = c.bloom(img, 210, 22, 0.3)
        img = c.vignette(img, 0.4)
        sx = sy = 0.0
        for ti, amp in ((self.slam_t, 26.0), (self.B[last]["key_t"] - 0.05, 12.0) if self.n > 1 else (99, 0)):
            d = t - ti
            if 0 <= d < 0.5:
                f = math.exp(-d * 9) * amp
                sx += f * math.sin(d * 2 * math.pi * 13)
                sy += f * math.cos(d * 2 * math.pi * 11) * 0.8
        if abs(sx) + abs(sy) > 0.5:
            M = np.float32([[1, 0, sx], [0, 1, sy]])
            img = cv2.warpAffine(img, M, (c.W, c.H), borderMode=cv2.BORDER_REFLECT)
        return c.grain(img, i, 2.6)
