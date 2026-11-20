"""/swiss-grid: Swiss International poster in motion: off-white paper, black grotesk, one red. The full-bleed shot springs into a grid that draws itself in, then on every keyword the black-and-white video cell reflows to a new module, huge words rise through masks, a red block wipes to its new place and a split-flap counter flips."""
import math

import cv2
import numpy as np

import core as c
import fx
from plan import Plan

PAPER = (241, 238, 231)
INK = (17, 17, 17)
RED = (226, 36, 22)
M, G = 54, 18
NC, NR = 6, 12
CW = (c.W - 2 * M - (NC - 1) * G) / NC
RH = (c.H - 2 * M - (NR - 1) * G) / NR
SHUT = 0.6


def mod(c0, r0, nc, nr):
    return (M + c0 * (CW + G), M + r0 * (RH + G), nc * CW + (nc - 1) * G, nr * RH + (nr - 1) * G)


# layouts: video cell, headline area, red block (+ which edge it wipes in from), caption spot
LAYOUTS = [
    {"vid": mod(1, 4, 5, 8), "txt": mod(0, 2, 6, 2), "red": mod(0, 4, 1, 2), "red_in": "l", "vert": mod(0, 6, 1, 6)},
    {"vid": mod(0, 0, 4, 8), "txt": mod(0, 8, 6, 4), "red": mod(4, 2, 2, 6), "red_in": "t", "vert": None},
    {"vid": mod(0, 4, 6, 5), "txt": mod(0, 2, 6, 2), "red": mod(0, 9, 6, 1), "red_in": "r", "vert": None},
]
HERO = {"vid": mod(0, 2, 6, 8), "txt": mod(0, 10, 6, 2), "red": mod(0, 0, 4, 2), "red_in": "b", "vert": None}
ENTRY = ["up", "left", "letters", "down"]


def mirror(L):
    f = lambda r: (c.W - r[0] - r[2], r[1], r[2], r[3]) if r else None
    return {"vid": f(L["vid"]), "txt": L["txt"], "red": f(L["red"]), "red_in": {"l": "r", "r": "l"}.get(L["red_in"], L["red_in"]),
            "vert": f(L["vert"])}


def clip_over(dst, patch, x, y, box, opacity=1.0):
    """over() restricted to box (x0, y0, x1, y1)."""
    h, w = patch.shape[:2]
    x, y = int(round(x)), int(round(y))
    bx0, by0, bx1, by1 = (int(round(v)) for v in box)
    X0, Y0 = max(x, bx0, 0), max(y, by0, 0)
    X1, Y1 = min(x + w, bx1, dst.shape[1]), min(y + h, by1, dst.shape[0])
    if X0 >= X1 or Y0 >= Y1:
        return
    c.over(dst, patch[Y0 - y:Y1 - y, X0 - x:X1 - x], X0, Y0, opacity)


def rect_fill(dst, x, y, w, h, color, a=1.0):
    x0, y0, x1, y1 = int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, dst.shape[1]), min(y1, dst.shape[0])
    if x1 > x0 and y1 > y0:
        dst[y0:y1, x0:x1] = dst[y0:y1, x0:x1] * (1 - a) + np.float32(color) * a


def mono(img, mix=1.0):
    g = img @ np.float32([0.299, 0.587, 0.114]) / 255
    g = np.clip((g - 0.42) * 1.38 + 0.5, 0, 1)[..., None]
    out = np.float32(INK) + (np.float32(PAPER) - np.float32(INK)) * g
    return out if mix >= 1 else img * (1 - mix) + out * mix


class Style:
    def __init__(self, clip, opts):
        self.clip = clip
        self.P = P = Plan(clip, opts)
        self.fr = c.Framing(clip, 1215)
        self.B = B = P.beats
        n = len(B)
        self.lay = []
        for k in range(n):
            if k == n - 1:
                self.lay.append(HERO)
            else:
                L = LAYOUTS[k % 3]
                self.lay.append(mirror(L) if (k // 3) % 2 else L)
        self.sw = [0.0] + [max(B[k]["key_t"] - 0.12, B[k - 1]["key_t"] + 0.3) for k in range(1, n)]
        self.dur = min(clip.n, len(clip.face)) / clip.fps
        rng = np.random.default_rng(3)
        base = np.empty((c.H, c.W, 3), np.float32)
        base[:] = PAPER
        blot = cv2.resize(cv2.GaussianBlur(rng.normal(0, 1, (c.H // 8, c.W // 8)).astype(np.float32), (0, 0), 3),
                          (c.W, c.H))
        self.paper = base + blot[..., None] * 5 + rng.normal(0, 2.2, (c.H, c.W, 1)).astype(np.float32)
        self.heads = [self._headline(k) for k in range(n)]
        self._tx = {}
        self.tiles = self._tiles()
        # grid guides: x of every column edge, y of every row edge
        self.vx = sorted([M + k * (CW + G) for k in range(NC)] + [M + k * (CW + G) + CW for k in range(NC)])
        self.hy = sorted([M + r * (RH + G) for r in range(NR)] + [M + r * (RH + G) + RH for r in range(NR)])

    # ---------- type ----------
    def tx(self, s, key, size, color=INK, wght=None, tracking=0.0):
        kk = (s, key, size, color, wght, tracking)
        if kk not in self._tx:
            self._tx[kk] = c.text_patch(s, key, size, color, wght, tracking)
        return self._tx[kk]

    def _headline(self, k):
        b, L = self.B[k], self.lay[k]
        x, y, w, h = L["txt"]
        words = b["label"].split()
        if len(words) > 1 and h > 400:
            mid = max(1, min(len(words) - 1, round(len(words) / 2)))
            lines = [" ".join(words[:mid]), " ".join(words[mid:])]
        else:
            lines = [b["label"]]
        tr = -0.035
        size = min(c.fit_size(s, "grotesk", w - 8, 820, tr) for s in lines)
        size = int(min(size, h / (len(lines) * 0.98 + 0.12), 300))
        color = RED if k == len(self.B) - 1 else INK
        out = []
        lead = size * 0.95
        top = y + (h - (len(lines) - 1) * lead - size * 0.76) / 2 + size * 0.74
        for j, s in enumerate(lines):
            p, base = c.text_patch(s, "grotesk", size, color, 820, tr)
            f = c.font("grotesk", size, 820)
            xs = [4.0]
            for ch in s:
                xs.append(xs[-1] + f.getlength(ch) + tr * size)
            out.append({"s": s, "p": p, "base": base, "x": x - 4, "bl": top + j * lead, "size": size, "xs": xs})
        return out

    def _tiles(self):
        tw, th = int(round(CW)), int(round(2 * RH + G))
        hh = th // 2
        out = {}
        for ch in "0123456789 ":
            t = c.fill(c.rrect(tw, th, 10), INK)
            if ch != " ":
                p, base = c.text_patch(ch, "grotesk", 236, PAPER, 640, 0.0)
                c.over(t, p, (tw - p.shape[1]) / 2 + 3, th / 2 - base + 236 * 0.36)
            t[hh - 2:hh + 2, :, :3] *= 0.25
            t[hh - 2:hh + 2, :, :3] += np.float32(PAPER) * 0.2 * t[hh - 2:hh + 2, :, 3:4] / 255
            top = np.zeros_like(t)
            top[:hh] = t[:hh]
            bot = np.zeros_like(t)
            bot[hh:] = t[hh:]
            out[ch] = (t, top, bot)
        return out

    # ---------- timing ----------
    def beat_at(self, t):
        k = 0
        for j, s in enumerate(self.sw):
            if t >= s:
                k = j
        return k

    def cell_rect(self, t):
        cur = np.float32([0, 0, c.W, c.H])
        for k, s in enumerate(self.sw):
            if t < s:
                break
            e = fx.spring(t, s, 1.45, 0.72) if k == 0 else fx.spring(t, s, 1.9, 0.62)
            cur = cur + (np.float32(self.lay[k]["vid"]) - cur) * e
        return cur

    # ---------- pieces ----------
    def guides(self, img, t):
        a = c.lerp(0.9, 0.14, fx.smooth(t, 0.42, 0.55))
        order = sorted(range(len(self.vx)), key=lambda j: abs(self.vx[j] - c.W / 2))
        for r, j in enumerate(order):
            p = fx.expo(t, 0.05 + 0.016 * r, 0.34)
            if p <= 0:
                continue
            x = int(round(self.vx[j]))
            L = int(c.H * p)
            if j % 2:
                rect_fill(img, x, c.H - L, 2, L, INK, a)
            else:
                rect_fill(img, x, 0, 2, L, INK, a)
        for r, y in enumerate(self.hy):
            p = fx.expo(t, 0.1 + 0.009 * r, 0.32)
            if p <= 0:
                continue
            L = int(c.W * p)
            if r % 2:
                rect_fill(img, c.W - L, y, L, 2, INK, a * 0.75)
            else:
                rect_fill(img, 0, y, L, 2, INK, a * 0.75)

    def red(self, img, t, k):
        """The red block: wipes out of the old module and into the new one."""
        dirs = {"l": (1, 0), "r": (-1, 0), "t": (0, 1), "b": (0, -1)}
        for j in (k - 1, k):
            if j < 0:
                continue
            x, y, w, h = self.lay[j]["red"]
            if j == k:
                p = fx.expo(t, self.sw[k] + (0.18 if k == 0 else 0.06), 0.42)
                side = self.lay[k]["red_in"]
            else:
                p = 1 - fx.smooth(t, self.sw[k], 0.22)
                side = {"l": "r", "r": "l", "t": "b", "b": "t"}[self.lay[j]["red_in"]]
            if p <= 0.002:
                continue
            if side == "l":
                rect_fill(img, x, y, w * p, h, RED)
            elif side == "r":
                rect_fill(img, x + w * (1 - p), y, w * p, h, RED)
            elif side == "t":
                rect_fill(img, x, y, w, h * p, RED)
            else:
                rect_fill(img, x, y + h * (1 - p), w, h * p, RED)
            # caption knocked out of the red block
            if j == k and p > 0.5:
                sub = self.B[k].get("sub", "") or self.B[k]["label"].upper()
                q = fx.expo(t, self.sw[k] + 0.3, 0.4)
                pp, base = self.tx(sub, "mono", 30, PAPER, 600, 0.14)
                box = (x, y, x + w, y + h)
                if w >= 600 and h >= 250:
                    sz = int(min(96, c.fit_size(sub, "grotesk", w - 60, 780, -0.02)))
                    pp, base = self.tx(sub, "grotesk", sz, PAPER, 780, -0.02)
                    clip_over(img, pp, x + 26, y + h - 30 - base + (1 - q) * (sz + 40), box)
                elif w > pp.shape[1] + 40:
                    clip_over(img, pp, x + 22, y + h - 24 - base + (1 - q) * 50, box)
                else:
                    rp = np.rot90(pp).copy()
                    clip_over(img, rp, x + 24 - (1 - q) * 50, y + h - rp.shape[0] - 22, box)

    def video(self, img, t, i, canvas):
        """The video cell, with real motion blur (temporal supersampling of the moving rect)."""
        r0 = self.cell_rect(t)
        r1 = self.cell_rect(t - 0.32 / self.clip.fps)
        speed = float(np.abs(r0 - r1).max())
        n = 1 if speed < 3 else int(min(6, 2 + speed / 16))
        fb = self.fr.face_box(i)
        k = self.beat_at(t)
        nxt = self.sw[k + 1] if k + 1 < len(self.sw) else self.dur + 1
        push = fx.smooth(t, self.sw[k] + 0.3, max(0.5, nxt - self.sw[k]))
        if k > 0:  # the previous beat's push eases out with the reflow
            pk = fx.smooth(self.sw[k], self.sw[k - 1] + 0.3, max(0.5, self.sw[k] - self.sw[k - 1]))
            push += pk * (1 - fx.spring(t, self.sw[k], 1.9, 0.62))
        zoom = 1.0 + 0.11 * push
        mix = fx.smooth(t, 0.06, 0.32)
        acc = None
        x0s, y0s, x1s, y1s = [], [], [], []
        rects = [r1 + (r0 - r1) * (s + 0.5) / n for s in range(n)] if n > 1 else [r0]
        for x, y, w, h in rects:
            x0s.append(x), y0s.append(y), x1s.append(x + w), y1s.append(y + h)
        X0, Y0 = int(math.floor(min(x0s))), int(math.floor(min(y0s)))
        X1, Y1 = int(math.ceil(max(x1s))), int(math.ceil(max(y1s)))
        acc = np.zeros((Y1 - Y0, X1 - X0, 4), np.float32)
        for x, y, w, h in rects:
            wi, hi = max(2, int(round(w))), max(2, int(round(h)))
            s = max(wi / c.W, hi / c.H) * zoom
            sw_, sh_ = wi / s, hi / s
            cx0 = c.clamp(fb[0] - sw_ * 0.5, 0, c.W - sw_)
            cy0 = c.clamp(fb[1] - sh_ * 0.36, 0, c.H - sh_)
            Mx = np.float32([[s, 0, -cx0 * s], [0, s, -cy0 * s]])
            crop = cv2.warpAffine(canvas, Mx, (wi, hi), flags=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REFLECT)
            crop = mono(crop, mix)
            patch = np.concatenate([crop, np.full((hi, wi, 1), 255, np.float32)], -1)
            ox, oy = int(round(x)) - X0, int(round(y)) - Y0
            if n == 1:
                c.over(acc, patch, ox, oy)
            else:
                self._add(acc, patch / n, ox, oy)
        c.over(img, acc, X0, Y0)
        return r0

    @staticmethod
    def _add(acc, p, x, y):
        h, w = p.shape[:2]
        x0, y0 = max(x, 0), max(y, 0)
        x1, y1 = min(x + w, acc.shape[1]), min(y + h, acc.shape[0])
        if x1 > x0 and y1 > y0:
            acc[y0:y1, x0:x1] += p[y0 - y:y1 - y, x0 - x:x1 - x]

    def headline(self, img, t, k):
        for j in (k - 1, k):
            if j < 0:
                continue
            lines = self.heads[j]
            mode = ENTRY[j % 4] if j < len(self.B) - 1 else "down"
            t_in = self.sw[j] + (0.24 if j == 0 else 0.1)
            t_out = self.sw[k] if j == k - 1 else 99
            tx0, _, tw, _ = self.lay[j]["txt"]
            for n, ln in enumerate(lines):
                p, size, bl = ln["p"], ln["size"], ln["bl"]
                box = (tx0 - 8, bl - 0.9 * size, tx0 + tw + 8, bl + 0.27 * size)
                bh = box[3] - box[1]

                def off(tt, d=0.0):
                    ein = fx.expo(tt, t_in + 0.07 * n + d, 0.5)
                    eout = fx.smooth(tt, t_out + 0.04 * n, 0.24)
                    if mode == "left":
                        return -(1 - ein) * (p.shape[1] + 60), -eout * bh
                    if mode == "down":
                        return 0.0, -(1 - ein) * bh - eout * bh
                    return 0.0, (1 - ein) * bh - eout * bh

                if mode == "letters":
                    xs = ln["xs"]
                    for q in range(len(ln["s"])):
                        a0, a1 = int(math.floor(xs[q])), int(math.ceil(xs[q + 1])) + 2
                        if q == 0:
                            a0 = 0
                        if q == len(ln["s"]) - 1:
                            a1 = p.shape[1]
                        if ln["s"][q] == " ":
                            continue
                        sl = p[:, a0:a1]
                        d = 0.028 * q
                        o0, o1 = off(t, d), off(t - SHUT / self.clip.fps, d)
                        if o0[1] >= bh - 1:
                            continue
                        bp, pad = fx.blur_patch(sl, 0, (o0[1] - o1[1]) * 0.9)
                        clip_over(img, bp, ln["x"] + a0 - pad, bl - ln["base"] + o0[1] - pad, box)
                    continue
                o0, o1 = off(t), off(t - SHUT / self.clip.fps)
                if abs(o0[1]) >= bh - 1 or o0[0] <= -(p.shape[1] + 50):
                    continue
                bp, pad = fx.blur_patch(p, (o0[0] - o1[0]) * 0.9, (o0[1] - o1[1]) * 0.9)
                clip_over(img, bp, ln["x"] + o0[0] - pad, bl - ln["base"] + o0[1] - pad, box)
            # red underline on the beat's second keyword
            b = self.B[j]
            if len(b["kw"]) > 1 and j == k:
                ln = lines[-1]
                e = fx.expo(t, b["kw"][1]["s"] - 0.04, 0.4)
                if e > 0:
                    rect_fill(img, ln["x"] + 6, ln["bl"] + ln["size"] * 0.1, (ln["p"].shape[1] - 14) * e,
                              max(10, ln["size"] * 0.07), RED)

    def counter(self, img, t, k):
        n = len(self.B)
        x0, y0, _, _ = mod(4, 0, 1, 2)
        x1, _, _, _ = mod(5, 0, 1, 2)
        tw, th = self.tiles["0"][0].shape[1], self.tiles["0"][0].shape[0]
        for col, X in enumerate((x0, x1)):
            seq = [(-1.0, " ")]
            for j in range(n):
                ch = f"{j + 1:02d}"[col]
                tj = (0.16 + 0.08 * col) if j == 0 else self.sw[j] + 0.04 + 0.06 * col
                if ch != seq[-1][1]:
                    seq.append((tj, ch))
            cur, prev, tc = seq[0][1], " ", -1.0
            for tj, ch in seq[1:]:
                if t >= tj:
                    prev, cur, tc = cur, ch, tj
            cx, cy = X + tw / 2, y0 + th / 2
            p = (t - tc) / 0.36
            if tc < 0 or p >= 1:
                c.over(img, self.tiles[cur][0], X, y0)
                continue
            c.over(img, self.tiles[cur][1], X, y0)
            c.over(img, self.tiles[prev][2], X, y0)
            if p < 0.5:
                a = 90 * (p / 0.5) ** 2
                fx.card3d(img, self.tiles[prev][1], cx, cy, rx=a, persp=900, shade=0.7)
            else:
                a = -90 * (1 - fx.spring(t, tc + 0.18, 3.2, 0.5))
                fx.card3d(img, self.tiles[cur][2], cx, cy, rx=min(a, 8), persp=900, shade=0.7)

    def info(self, img, t, k):
        x, y, w, h = mod(0, 0, 4, 2)
        e = fx.expo(t, 0.2, 0.6)
        rect_fill(img, x, y, w * e, 10, INK)
        n = len(self.B)
        kick = (self.P.kicker or self.P.title or "").upper()
        lines = [(kick, INK), (f"NO. {k + 1:02d} / {n:02d}", INK)]
        if self.lay[k]["red"][0] < c.W / 2 and self.lay[k]["red"][1] < 300:
            return  # the red block owns this corner
        for j, (s, col) in enumerate(lines):
            if not s:
                continue
            q = fx.expo(t, 0.3 + 0.06 * j, 0.5)
            pp, base = self.tx(s, "mono", 30, col, 560, 0.12)
            bl = y + 64 + j * 46
            clip_over(img, pp, x, bl - base + (1 - q) * 44, (x, bl - 34, x + w, bl + 12))
        title = self.P.title or ""
        if title:
            q = fx.expo(t, 0.42, 0.6)
            pp, base = self.tx(title, "grotesk", 58, INK, 760, -0.025)
            bl = y + h - 22
            clip_over(img, pp, x - 3, bl - base + (1 - q) * 70, (x - 4, bl - 60, x + w, bl + 18))

    def vertical(self, img, t, k):
        L = self.lay[k]
        if not L.get("vert"):
            return
        x, y, w, h = L["vert"]
        s = self.B[k]["text"].upper()
        pp, base = self.tx(s, "mono", 30, INK, 560, 0.1)
        rp = np.rot90(pp).copy()
        q = fx.expo(t, self.sw[k] + 0.35, 0.6) * (1 - fx.smooth(t, self.sw[k + 1] if k + 1 < len(self.sw) else 99, 0.2))
        if q <= 0:
            return
        clip_over(img, rp, x + 4, y + h - rp.shape[0] + (1 - q) * rp.shape[0] * 0.6, (x, y, x + w, y + h))

    def footer(self, img, t):
        e = fx.expo(t, 0.25, 0.6)
        y = c.H - M + 16
        rect_fill(img, M, y, (c.W - 2 * M) * e, 3, INK, 0.25)
        rect_fill(img, M, y, (c.W - 2 * M) * c.clamp(t / self.dur) * e, 3, INK)
        tc = f"{int(t // 60):02d}:{t % 60:05.2f}"
        pp, base = self.tx(tc, "mono", 28, INK, 520, 0.08)
        c.over(img, pp, c.W - M - pp.shape[1] + 4, y + 34 - base, e)
        hd = (self.P.handle or "").upper()
        if hd:
            pp, base = self.tx(hd, "mono", 28, INK, 520, 0.08)
            c.over(img, pp, M - 4, y + 34 - base, e)

    # ---------- frame ----------
    def frame(self, i, rgb, alpha):
        t = i / self.clip.fps
        k = self.beat_at(t)
        canvas, _ = self.fr.place(rgb, alpha)
        img = self.paper.copy()
        self.guides(img, t)
        self.red(img, t, k)
        self.info(img, t, k)
        self.counter(img, t, k)
        self.vertical(img, t, k)
        self.footer(img, t)
        self.video(img, t, i, canvas)
        self.headline(img, t, k)
        return c.grain(img, i, 1.6)
