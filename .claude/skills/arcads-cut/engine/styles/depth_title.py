"""/depth-title: cinematic depth headline. Each beat's word fills the frame width BEHIND the speaker: the first one rushes in from past the camera, the next ones slam in with a different 3D tilt each, gradient-lit, tracking tightening as they land, with real motion blur and the speaker's shadow falling on them; a small type layer slides with parallax in front while the camera pushes and drifts under a film grade."""
import math

import cv2
import numpy as np

import core as c
import fx
from plan import Plan

CREAM = (255, 247, 232)
AMBER = (236, 148, 66)
PERSP = 1800.0
SLAMS = ["tilt", "swing", "drop"]


def warp_card(dst, patch, cx, cy, rx=0.0, ry=0.0, rz=0.0, scale=1.0, z=0.0, opacity=1.0, persp=PERSP):
    """A flat premultiplied patch as a 3D card, warped straight into dst (clipped to the canvas, any size)."""
    h, w = patch.shape[:2]
    pts = np.float32([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0], [w / 2, h / 2, 0], [-w / 2, h / 2, 0]])
    quad, zz = fx.project(pts, cx, cy, rx, ry, rz, scale, z, persp)
    if (zz < 60).any():
        return quad
    H_, W_ = dst.shape[:2]
    x0, y0 = np.floor(quad.min(0)).astype(int) - 2
    x1, y1 = np.ceil(quad.max(0)).astype(int) + 2
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W_), min(y1, H_)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return quad
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    Mh = cv2.getPerspectiveTransform(src, (quad - np.float32([x0, y0])).astype(np.float32))
    out = cv2.warpPerspective(patch, Mh, (int(x1 - x0), int(y1 - y0)), flags=cv2.INTER_LINEAR, borderValue=0)
    c.over(dst, out, x0, y0, opacity)
    return quad


class Style:
    def __init__(self, clip, opts):
        self.clip = clip
        self.P = P = Plan(clip, opts)
        self.fr = fr = c.Framing(clip, 1215)
        self.B = B = P.beats
        self.n = n = len(B)
        self.dur = min(clip.n, len(clip.face)) / clip.fps
        fb = np.median(np.array([fr.face_box(i) for i in range(min(clip.n, len(clip.face)))]), 0)
        self.head_top = fb[1] - fb[3] * 0.62
        # when each word starts moving and when it lands
        self.ts = [0.0] + [B[k]["key_t"] - 0.16 for k in range(1, n)]
        dxs = [0, -34, 30, -22, 26]
        keys = [{"t": 0, "zoom": 1.04}, {"t": 0.3, "zoom": 1.11, "ease": "spring", "freq": 2.0, "damp": 0.5},
                {"t": 0.7, "zoom": 1.06, "ease": "smooth", "dur": 0.8}]
        for k in range(1, n):
            L = B[k]["key_t"]
            zk = min(1.12, 1.06 + 0.02 * k)
            dx = 0 if k == n - 1 else dxs[k % len(dxs)]
            keys += [{"t": L - 0.06, "zoom": zk + (0.06 if k == n - 1 else 0.035), "dx": dx, "ease": "spring",
                      "freq": 1.8, "damp": 0.6},
                     {"t": L + 0.3, "zoom": zk, "ease": "smooth", "dur": 0.9}]
        self.cam = fx.Cam(fr, keys, drift=6)
        self.words = [self._lines(b["label"]) for b in B]
        self._wp = {}
        self._tx = {}
        rng = np.random.default_rng(5)
        self.dust = [(rng.uniform(0, c.W), rng.uniform(0, c.H), rng.uniform(1.2, 3.2), rng.uniform(0.2, 0.7),
                      rng.uniform(0, 6.28)) for _ in range(46)]

    # ---------- type ----------
    def _lines(self, label):
        s = label.upper()
        ws = s.split()
        lines = [s]
        if len(ws) > 1:
            best = None
            for m in range(1, len(ws)):
                a, b = " ".join(ws[:m]), " ".join(ws[m:])
                score = abs(len(a) - len(b))
                if best is None or score < best[0]:
                    best = (score, [a, b])
            single = c.fit_size(s, "anton", 930, None, -0.01)
            if single < 300:
                lines = best[1]
        sizes = [min(520, c.fit_size(ln, "anton", 930, None, -0.01)) for ln in lines]
        tot = sum(sz * 1.0 for sz in sizes)
        cap = 440.0
        if tot > cap:
            sizes = [int(sz * cap / tot) for sz in sizes]
        return list(zip(lines, sizes))

    def word_patch(self, k, tracking):
        tq = round(tracking, 3)
        kk = (k, tq)
        if kk in self._wp:
            return self._wp[kk]
        ps = []
        for ln, sz in self.words[k]:
            p, base = c.text_patch(ln, "anton", sz, (255, 255, 255), None, tq, 4)
            ys = np.nonzero(p[..., 3].max(1) > 8)[0]
            ps.append(p[max(0, ys[0] - 2):ys[-1] + 3] if len(ys) else p)
        gap = 18
        w = max(p.shape[1] for p in ps)
        h = sum(p.shape[0] for p in ps) + gap * (len(ps) - 1)
        out = np.zeros((h, w, 4), np.float32)
        y = 0
        for p in ps:
            c.over(out, p, (w - p.shape[1]) / 2, y)
            y += p.shape[0] + gap
        u = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
        col = np.float32(CREAM) * (1 - u ** 1.3) + np.float32(AMBER) * u ** 1.3
        fade = (1 - 0.35 * np.clip((u - 0.55) / 0.45, 0, 1))
        a = out[..., 3:4] / 255 * fade
        res = np.concatenate([col * a, a * 255], -1)
        if len(self._wp) > 400:
            self._wp.clear()
        self._wp[kk] = res
        return res

    def tx(self, s, key, size, color, wght=None, tracking=0.0):
        kk = (s, key, size, color, wght, tracking)
        if kk not in self._tx:
            self._tx[kk] = c.text_patch(s, key, size, color, wght, tracking)
        return self._tx[kk]

    # ---------- word motion ----------
    def word_cy(self, k):
        """Vertical centre of word k: the head cuts into the lower part of the (last) line, never the whole line."""
        p = self.word_patch(k, 0.0)
        last = self.words[k][-1][1] * 0.74
        return self.head_top + 0.4 * last - p.shape[0] / 2

    def pose(self, k, t, camdx, camdy):
        """3D pose of word k at time t (None when not visible)."""
        n = self.n
        ts = self.ts[k]
        if t < ts:
            return None
        nxt = self.ts[k + 1] if k + 1 < n else 99
        p = self.word_patch(k, 0.0)
        cy = self.word_cy(k)
        cx = c.W / 2 + camdx * 0.4
        cy = cy + camdy * 0.4
        rx = ry = rz = z = 0.0
        sc, op = 1.0, 1.0
        land = ts + 0.16
        if k == 0:  # rushes in from past the camera
            e = fx.expo(t, 0.0, 0.62)
            z = -1380 * (1 - e)
            rx = 10 * (1 - e)
            trk = c.lerp(0.42, -0.01, fx.expo(t, 0.0, 0.7))
        elif k == n - 1:  # from deep behind, punches forward past its mark and settles
            e = fx.spring(t, ts, 1.55, 0.64)
            z = 2100 * (1 - e)
            op = c.clamp((t - ts) / 0.08)
            trk = c.lerp(0.5, -0.01, fx.expo(t, ts, 0.7))
        else:
            mode = SLAMS[(k - 1) % 3]
            e = fx.spring(t, ts, 1.65, 0.52)
            op = c.clamp((t - ts) / 0.06)
            trk = c.lerp(0.32, -0.01, fx.expo(t, ts, 0.6))
            if mode == "tilt":
                rx = -78 * (1 - e)
                z = 500 * (1 - e)
                cy -= 120 * (1 - e)
            elif mode == "swing":
                ry = 72 * (1 - e)
                cx += 260 * (1 - e)
                z = 300 * (1 - e)
            else:
                cy -= 700 * (1 - e)
                rx = 35 * (1 - e)
                rz = -6 * (1 - e)
        # slow creep toward camera and tracking tightening while it holds
        creep = fx.smooth(t, land, max(0.6, min(nxt, self.dur) - land))
        sc *= 1 + 0.03 * creep
        trk -= 0.012 * creep
        # exit: falls back into depth
        if t >= nxt:
            q = fx.smooth(t, nxt, 0.26)
            z += 900 * q
            op *= 1 - q
            rx += 8 * q
            if op <= 0.01:
                return None
        return dict(cx=cx, cy=cy, rx=rx, ry=ry, rz=rz, z=z, sc=sc, op=op, trk=trk)

    def draw_word(self, layer, k, t, camdx, camdy, front_layer=None):
        P0 = self.pose(k, t, camdx, camdy)
        if P0 is None:
            return
        P1 = self.pose(k, t - 0.55 / self.clip.fps, camdx, camdy) or P0
        patch = self.word_patch(k, P0["trk"])
        h, w = patch.shape[:2]
        pts = np.float32([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0], [w / 2, h / 2, 0], [-w / 2, h / 2, 0]])
        q0, _ = fx.project(pts, P0["cx"], P0["cy"], P0["rx"], P0["ry"], P0["rz"], P0["sc"], P0["z"], PERSP)
        q1, _ = fx.project(pts, P1["cx"], P1["cy"], P1["rx"], P1["ry"], P1["rz"], P1["sc"], P1["z"], PERSP)
        disp = float(np.abs(q0 - q1).max())
        n = 1 if disp < 4 else int(min(9, 2 + disp / 18))
        s_eff = PERSP / max(PERSP + P0["z"], 1) * P0["sc"]
        wf = c.clamp((s_eff - 1.5) / 1.4) if front_layer is not None else 0.0
        tmp = layer if n == 1 and wf == 0 else np.zeros_like(layer)
        for j in range(n):
            u = (j + 0.5) / n if n > 1 else 1.0
            Pm = {kk: P1[kk] + (P0[kk] - P1[kk]) * u for kk in P0}
            if n == 1:
                warp_card(tmp, patch, Pm["cx"], Pm["cy"], Pm["rx"], Pm["ry"], Pm["rz"], Pm["sc"], Pm["z"], P0["op"])
            else:
                self._acc(tmp, patch, Pm, P0["op"] / n)
        if tmp is layer:
            return
        if disp > 60:
            tmp = cv2.GaussianBlur(tmp, (0, 0), min(10, disp / 30))
        if wf > 0:
            c.over(front_layer, tmp * (wf * 0.8))
        c.over(layer, tmp * (1 - wf * 0.9) if wf else tmp)

    @staticmethod
    def _acc(dst, patch, Pm, op):
        tmp = np.zeros_like(dst)
        warp_card(tmp, patch, Pm["cx"], Pm["cy"], Pm["rx"], Pm["ry"], Pm["rz"], Pm["sc"], Pm["z"], 1.0)
        dst += tmp * op

    # ---------- front type (parallax) ----------
    def front(self, layer, t, k, camdx, camdy):
        n = self.n
        b = self.B[k]
        par = 1.7
        y = 1600 + camdy * par
        e = fx.expo(t, 0.5, 0.7)
        # rules either side of the sub line
        for j in (k - 1, k):
            if j < 0:
                continue
            bj = self.B[j]
            s = f"{j + 1:02d}   {(bj.get('sub') or bj['label']).upper()}"
            pp, base = self.tx(s, "mono", 34, CREAM, 560, 0.32)
            tin = (0.5 if j == 0 else bj["key_t"] + 0.05)
            ein = fx.expo(t, tin, 0.55)
            eout = fx.smooth(t, self.B[j + 1]["key_t"] - 0.1, 0.25) if j + 1 < n else 0.0
            if j == k - 1 and eout <= 0:
                continue
            def xat(tt):
                ei = fx.expo(tt, tin, 0.55)
                eo = fx.smooth(tt, self.B[j + 1]["key_t"] - 0.1, 0.25) if j + 1 < n else 0.0
                return (1 - ei) * 160 - eo * 220
            x = c.W / 2 + camdx * par + xat(t)
            mvx = (xat(t) - xat(t - 0.022)) * 0.8
            fx.put(layer, pp, x, y - base + pp.shape[0] / 2, opacity=ein * (1 - eout), mv=(mvx, 0))
        wl = 260 * e
        for side in (-1, 1):
            x0 = c.W / 2 + camdx * par + side * 330
            a = np.ones((2, max(1, int(wl))), np.float32)
            p = c.fill(a, CREAM, 0.7)
            c.over(layer, p, x0 if side > 0 else x0 - wl, y - 12)
        # kicker above the headline (mid layer)
        kick = (self.P.kicker or self.P.title or "").upper()
        if kick:
            pp, base = self.tx(kick, "mono", 30, CREAM, 520, 0.42)
            q = fx.expo(t, 0.55, 0.8)
            for j in range(self.n):  # step aside while a tall (two-line) word is up
                if len(self.words[j]) > 1:
                    t0 = self.ts[j] - 0.1
                    t1 = self.ts[j + 1] if j + 1 < self.n else 99
                    q *= 1 - fx.smooth(t, t0, 0.2) * (1 - fx.smooth(t, t1, 0.3))
            fx.put(layer, pp, c.W / 2 + camdx * 0.9, 92 + camdy * 0.9 + (1 - q) * 30, opacity=q * 0.85)
        # counter at the bottom corner
        pp, base = self.tx(f"{k + 1:02d}/{n:02d}", "mono", 30, CREAM, 520, 0.2)
        c.over(layer, pp, 64 + camdx * par, 1780 + camdy * par, e * 0.8)

    def dust_layer(self, img, t, camdx, camdy):
        for x, y, r, a, ph in self.dust:
            xx = (x + camdx * 1.9 + 14 * math.sin(t * 0.7 + ph)) % c.W
            yy = (y - t * 22 * r + camdy * 1.9) % c.H
            g = math.exp(-((t * 1.3 + ph) % 3 - 1.5) ** 2) * a
            s = int(r * 4) | 1
            yy0, xx0 = int(yy) - s, int(xx) - s
            if 0 <= yy0 and yy0 + 2 * s < c.H and 0 <= xx0 and xx0 + 2 * s < c.W:
                ys, xs = fx.grid(2 * s, 2 * s)
                m = np.exp(-((xs - s) ** 2 + (ys - s) ** 2) / (2 * r * r)) * g * 60
                img[yy0:yy0 + 2 * s, xx0:xx0 + 2 * s] += m[..., None] * np.float32([1.0, 0.85, 0.6])

    # ---------- frame ----------
    def frame(self, i, rgb, alpha):
        t = i / self.clip.fps
        img, a = self.cam.place(i, t, rgb, alpha)
        cp = self.cam.at(t)
        camdx, camdy = cp["dx"], cp["dy"]
        k = 0
        for j, s in enumerate(self.ts):
            if t >= s:
                k = j
        back = c.layer()
        frontw = c.layer()
        for j in (k - 1, k):
            if j >= 0:
                self.draw_word(back, j, t, camdx, camdy, frontw if j == 0 else None)
        # soft shadow of the word on the wall, the speaker's shadow on the word
        small = cv2.resize(back[..., 3], (c.W // 4, c.H // 4), interpolation=cv2.INTER_AREA)
        sh = cv2.resize(cv2.GaussianBlur(small, (0, 0), 6), (c.W, c.H))
        sh = np.roll(sh, 26, 0) / 255 * 0.42
        img *= (1 - sh[..., None])
        am = cv2.resize(cv2.GaussianBlur(cv2.resize(a, (c.W // 4, c.H // 4)), (0, 0), 5), (c.W, c.H))
        am = np.roll(np.roll(am, 30, 0), 18, 1)
        back *= (1 - 0.5 * am)[..., None]
        img = c.composite(img, a, back=back)
        img = img * (1 - frontw[..., 3:4] / 255) + frontw[..., :3]
        # payoff: an anamorphic streak where the last word lands
        L = self.B[-1]["key_t"]
        fa = math.exp(-((t - (L + 0.06)) / 0.12) ** 2) if self.n > 1 else 0.0
        if fa > 0.02:
            yy = self.word_cy(self.n - 1) + camdy * 0.4
            fx.flare(img, c.W / 2 + camdx * 0.4, yy, length=1700, thick=9, color=(255, 170, 90), amount=fa * 1.1)
        self.dust_layer(img, t, camdx, camdy)
        fl = c.layer()
        self.front(fl, t, k, camdx, camdy)
        img = img * (1 - fl[..., 3:4] / 255) + fl[..., :3]
        # film grade: warm highlights, cool shadows, soft bloom, vignette, grain
        lum = (img @ np.float32([0.299, 0.587, 0.114]))[..., None] / 255
        img = img + (1 - lum) ** 2 * np.float32([-6, 1, 9]) + lum ** 2 * np.float32([9, 3, -7])
        img = c.grade(img, contrast=1.08, sat=0.94, lift=0.012)
        img = c.bloom(img, 205, 24, 0.3)
        img = c.vignette(img, 0.42)
        return c.grain(img, i, 3.2)
