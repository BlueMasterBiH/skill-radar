"""/cutout-grid: pop-art grid. The cut-out speaker on a bold colour field explodes into a 2x2 grid in a staggered 3D flip wave, splits into nine duotone tiles with a type tile in the middle, the rows slide past each other like a slot machine to deal the next word, then the centre tile swallows the grid back into one big hero with the payoff label."""
import math

import cv2
import numpy as np

import core as c
import fx
from plan import Plan

GUT = 14
INK = (14, 14, 16)
CARD = (246, 243, 236)
PAL = [((255, 72, 132), (34, 16, 70), (255, 238, 130)),
       ((36, 98, 255), (8, 12, 36), (130, 255, 214)),
       ((255, 204, 0), (56, 18, 88), (255, 128, 176)),
       ((0, 194, 158), (14, 28, 58), (255, 222, 200)),
       ((255, 104, 36), (28, 14, 58), (150, 224, 255)),
       ((138, 76, 255), (18, 10, 40), (255, 212, 96)),
       ((238, 40, 58), (18, 18, 28), (255, 240, 220)),
       ((168, 234, 40), (18, 40, 30), (250, 92, 172)),
       ((0, 184, 255), (10, 18, 50), (255, 250, 232))]
HERO0 = ((36, 98, 255), None, None)
HERO1 = ((255, 204, 0), None, None)


def grid_rect(g, r, cc):
    if g == 1:
        return (0.0, 0.0, float(c.W), float(c.H))
    tw = (c.W - (g + 1) * GUT) / g
    th = (c.H - (g + 1) * GUT) / g
    return (GUT + cc * (tw + GUT), GUT + r * (th + GUT), tw, th)


def lerp_rect(a, b, e):
    return tuple(x + (y - x) * e for x, y in zip(a, b))


def shift(m, dx, dy):
    out = np.zeros_like(m)
    h, w = m.shape[:2]
    out[max(dy, 0):h + min(dy, 0), max(dx, 0):w + min(dx, 0)] = m[max(-dy, 0):h - max(dy, 0), max(-dx, 0):w - max(dx, 0)]
    return out


class Style:
    def __init__(self, clip, opts):
        self.clip = clip
        self.P = P = Plan(clip, opts)
        self.fr = c.Framing(clip, 1215)
        self.B = B = P.beats
        self.n = n = len(B)
        self.N = min(clip.n, len(clip.face))
        self.dur = self.N / clip.fps
        # layout per beat: opening 2x2, then 3x3 split, then row shuffles, last = hero
        seq = []
        for k in range(n):
            if k == n - 1 and n > 1:
                seq.append(("hero", 0))
            elif k == 0:
                seq.append(("g2", 0))
            elif k == 1:
                seq.append(("g3", 0))
            else:
                seq.append(("shuf", k - 1))
        if n == 1:
            seq = [("g2", 0)]
        self.seq = seq
        self.T = [0.04] + [B[k]["key_t"] - 0.16 for k in range(1, n)]
        # pop-art dots (big at the bottom), resized per tile
        ys, xs = np.mgrid[0:c.H, 0:c.W].astype(np.float32)
        sp = 22.0
        u, v = (xs + ys * 0.5) % sp - sp / 2, (ys * 0.866) % sp - sp / 2
        rad = 1.2 + 6.0 * (ys / c.H) ** 1.6
        self.dots = np.clip(rad - np.hypot(u, v), 0, 1).astype(np.float32)
        self._tx = {}

    def tx(self, s, key, size, color, wght=None, tracking=0.0):
        kk = (s, key, size, color, wght, tracking)
        if kk not in self._tx:
            self._tx[kk] = c.text_patch(s, key, size, color, wght, tracking)
        return self._tx[kk]

    # ---------- tiles ----------
    def tile(self, w, h, pal, zoom=1.06, natural=False, rad=18):
        w, h = int(round(w)), int(round(h))
        if w < 6 or h < 6:
            return None
        fb = self.fb
        s = max(w / c.W, h / c.H) * zoom
        sw, sh = w / s, h / s
        x0 = c.clamp(fb[0] - sw * 0.5, 0, c.W - sw)
        y0 = c.clamp(fb[1] - sh * 0.36, 0, c.H - sh)
        M = np.float32([[s, 0, -x0 * s], [0, s, -y0 * s]])
        it = cv2.INTER_AREA if s < 0.8 else cv2.INTER_LINEAR
        rgb = cv2.warpAffine(self.canvas, M, (w, h), flags=it, borderMode=cv2.BORDER_REPLICATE)
        a = cv2.warpAffine(self.matte, M, (w, h), flags=it, borderMode=cv2.BORDER_REPLICATE)
        bg = np.float32(pal[0])
        out = np.empty((h, w, 3), np.float32)
        out[:] = bg
        d = cv2.resize(self.dots, (w, h), interpolation=cv2.INTER_AREA)
        out *= (1 - 0.16 * d)[..., None]
        off = max(2, int(w * 0.03))
        shd = shift(a, off, int(off * 0.8))
        out *= (1 - 0.42 * shd)[..., None]
        if natural:
            p = c.grade(rgb, contrast=1.1, sat=1.12, warm=0.01)
        else:
            g = rgb @ np.float32([0.299, 0.587, 0.114]) / 255
            g = np.clip((g - 0.3) * 1.75 + 0.42, 0, 1)
            g = g * g * (3 - 2 * g)
            p = np.float32(pal[1]) + (np.float32(pal[2]) - np.float32(pal[1])) * g[..., None]
        out = out * (1 - a[..., None]) + p * a[..., None]
        m = c.rrect(w, h, rad) if rad else np.ones((h, w), np.float32)
        return np.concatenate([out * m[..., None], m[..., None] * 255], -1)

    def type_tile(self, w, h, k, rad=18):
        w, h = int(round(w)), int(round(h))
        if w < 6 or h < 6:
            return None
        p = c.fill(c.rrect(w, h, rad), CARD)
        b = self.B[k]
        words = b["label"].upper().split()
        lines = words if len(words) <= 3 else [" ".join(words[:len(words) // 2]), " ".join(words[len(words) // 2:])]
        size = min(min(c.fit_size(s, "anton", w - 40) for s in lines), int(h * 0.7 / len(lines)), 150)
        y = h / 2 - (len(lines) * size * 1.0) / 2 + size * 0.86
        for s in lines:
            tp, base = self.tx(s, "anton", size, INK)
            c.over(p, tp, (w - tp.shape[1]) / 2, y - base)
            y += size * 1.0
        sub = (b.get("sub") or "").upper()
        if sub and w > 200:
            ss = int(min(30, c.fit_size(sub, "mono", w - 44, 560, 0.12)))
            if ss >= 22:
                tp, base = self.tx(sub, "mono", ss, (226, 36, 60), 560, 0.12)
                c.over(p, tp, (w - tp.shape[1]) / 2, h - 40 - base)
        pal = PAL[(k * 3) % 9][0]
        bar = c.fill(np.ones((10, w), np.float32), pal)
        c.over(p, bar * c.rrect(w, h, rad)[:10, :, None], 0, 0)
        return p

    CORNER = {0: (0, 0), 1: (0, 2), 2: (2, 0), 3: (2, 2)}

    def pal_g2(self, q):
        return self.pal_g3(*self.CORNER[q % 4])

    def idx_g2(self, q):
        r, cc = self.CORNER[q % 4]
        return r * 3 + cc

    def pal_g3(self, r, v):
        return PAL[(r * 4 + v * 7 + 5) % 9]

    # ---------- layouts ----------
    def zoom_at(self, t, idx):
        return 1.05 + 0.035 * math.sin(t * 1.4 + idx * 1.7)

    def draw_tile(self, img, rect, pal, t, idx, mv=(0.0, 0.0), natural=False, k_type=None, rad=18, scale=1.0):
        x, y, w, h = rect
        if k_type is not None:
            p = self.type_tile(w, h, k_type, rad)
        else:
            p = self.tile(w, h, pal, self.zoom_at(t, idx), natural, rad)
        if p is None:
            return
        fx.put(img, p, x + w / 2, y + h / 2, scale=scale, mv=mv)

    def g2(self, img, t, label_k):
        T0 = self.T[0]
        hero = None
        for q in range(4):
            r, cc = divmod(q, 2)
            dst = grid_rect(2, r, cc)
            st = T0 + 0.075 * (r + cc)
            e = fx.spring(t, st, 2.2, 0.7)
            if e >= 0.999 and t > st + 0.6:
                self.draw_tile(img, dst, self.pal_g2(q), t, self.idx_g2(q))
                continue
            src = (cc * c.W / 2, r * c.H / 2, c.W / 2, c.H / 2)
            gap = fx.smooth(t, st, 0.35)
            x, y, w, h = lerp_rect(src, dst, gap)
            wi, hi = int(round(w)), int(round(h))
            if hero is None:
                hero = self.tile(c.W, c.H, HERO0, 1.0, natural=True, rad=0)
            front = cv2.resize(hero[int(src[1]):int(src[1] + src[3]), int(src[0]):int(src[0] + src[2])], (wi, hi),
                               interpolation=cv2.INTER_AREA)
            front = front * c.rrect(wi, hi, int(18 * gap))[..., None]
            back = self.tile(wi, hi, self.pal_g2(q), self.zoom_at(t, self.idx_g2(q)))
            ry = 180 * e
            z = -320 * math.sin(math.pi * c.clamp(e))
            fx.card3d(img, front, x + w / 2, y + h / 2, ry=ry * (1 if cc else -1), z=z, persp=1600, shade=0.18,
                      back=back, shadow=0.35 * math.sin(math.pi * c.clamp(e)))
        self.banner(img, t, label_k)

    def banner(self, img, t, k):
        """Beat label on a black bar across the middle gutter."""
        b = self.B[k]
        L = b["key_t"] - 0.06
        out_t = self.T[k + 1] if k + 1 < self.n else 99
        e = fx.spring(t, L, 2.0, 0.62) * (1 - fx.smooth(t, out_t, 0.18))
        if e <= 0.01:
            return
        s = b["label"].upper()
        size = int(min(110, c.fit_size(s, "archivo", 860)))
        tp, base = self.tx(s, "archivo", size, (255, 255, 255))
        bw, bh = tp.shape[1] + 90, int(size * 1.5)
        cy = c.H / 2
        w = max(2, int(bw * min(e, 1.08)))
        bar = c.fill(c.rrect(w, bh, bh // 2), INK)
        fx.put(img, bar, c.W / 2, cy, shadow=0.45, sblur=18, sdy=10)
        q = fx.expo(t, L + 0.08, 0.4)
        box = (c.W / 2 - w / 2 + 10, cy - bh / 2, c.W / 2 + w / 2 - 10, cy + bh / 2)
        ty = cy + size * 0.36 - base + (1 - q) * bh
        h_, w_ = tp.shape[:2]
        x = c.W / 2 - w_ / 2
        X0, Y0 = int(max(x, box[0])), int(max(ty, box[1]))
        X1, Y1 = int(min(x + w_, box[2])), int(min(ty + h_, box[3]))
        if X1 > X0 and Y1 > Y0:
            c.over(img, tp[Y0 - int(ty):Y1 - int(ty), X0 - int(x):X1 - int(x)], X0, Y0)
        sub = (b.get("sub") or "").upper()
        if sub:
            sp, sb = self.tx(sub, "mono", 30, INK, 600, 0.18)
            pill = c.fill(c.rrect(sp.shape[1] + 44, 54, 27), (255, 255, 255))
            c.over(pill, sp, 22, 27 - sb + 11)
            qq = fx.spring(t, L + 0.14, 2.2, 0.6) * (1 - fx.smooth(t, out_t, 0.15))
            fx.put(img, pill, c.W / 2, cy + bh / 2 + 44, scale=max(0.01, qq), opacity=min(1, qq * 2))

    def g3_split(self, img, t, k):
        T = self.T[k]
        corners = self.CORNER
        dt = 0.6 / self.clip.fps
        # the four 2x2 tiles shrink into the corners
        for q in range(4):
            r, cc = divmod(q, 2)
            src, dst = grid_rect(2, r, cc), grid_rect(3, *corners[q])

            def rect(tt):
                return lerp_rect(src, dst, fx.spring(tt, T + 0.02 * q, 1.8, 0.7))
            R0, R1 = rect(t), rect(t - dt)
            mv = ((R0[0] + R0[2] / 2) - (R1[0] + R1[2] / 2), (R0[1] + R0[3] / 2) - (R1[1] + R1[3] / 2))
            self.draw_tile(img, R0, self.pal_g2(q), t, self.idx_g2(q), mv)
        # five new tiles slide in from the edges; the centre is the type tile
        news = [((0, 1), (0, -1)), ((1, 0), (-1, 0)), ((1, 2), (1, 0)), ((2, 1), (0, 1))]
        for j, ((r, cc), (dx, dy)) in enumerate(news):
            dst = grid_rect(3, r, cc)

            far = {(0, -1): dst[1] + dst[3] + 30, (-1, 0): dst[0] + dst[2] + 30, (1, 0): c.W - dst[0] + 30,
                   (0, 1): c.H - dst[1] + 30}[(dx, dy)]

            def rect2(tt):
                e = fx.spring(tt, T + 0.02 + 0.03 * j, 2.0, 0.72)
                return (dst[0] + dx * (1 - e) * far, dst[1] + dy * (1 - e) * far, dst[2], dst[3])
            R0, R1 = rect2(t), rect2(t - dt)
            if t < T + 0.02 + 0.03 * j:
                continue
            self.draw_tile(img, R0, self.pal_g3(r, cc), t, r * 3 + cc, (R0[0] - R1[0], R0[1] - R1[1]))
        e = fx.spring(t, T + 0.06, 2.2, 0.55)
        if e > 0.01:
            self.draw_tile(img, grid_rect(3, 1, 1), None, t, 4, k_type=k, scale=max(0.02, e))

    def shuf_state(self, k):
        """Row offsets (virtual index shift) after shuffle k (0 = the split grid)."""
        dirs = (1, -1, 1)
        steps = (1, 2, 1)
        reps = sum(1 for kk in range(1, k + 1) if self.seq[kk][0] == "shuf")
        return [dirs[r] * steps[r] * reps for r in range(3)]

    def g3_static(self, img, t, k, offs, prog=None):
        """3x3 grid with each row shifted by offs[r] tile steps (fractional while sliding)."""
        tw = grid_rect(3, 0, 0)[2] + GUT
        type_v = 1 - round(offs[1]) if prog is None else None
        for r in range(3):
            o = offs[r]
            vmin, vmax = int(math.floor(-o)) - 1, int(math.ceil(3 - o)) + 1
            for v in range(vmin, vmax):
                x, y, w, h = grid_rect(3, r, 0)
                xx = x + (v + o) * tw
                if xx > c.W or xx + w < 0:
                    continue
                mv = (0.0, 0.0)
                if prog is not None:
                    mv = (prog["vel"][r], 0.0)
                kt = None
                if r == 1:
                    if prog is None and v == type_v:
                        kt = k
                    elif prog is not None and v in prog["types"]:
                        kt = prog["types"][v]
                self.draw_tile(img, (xx, y, w, h), self.pal_g3(r, v), t, r * 3 + (v % 3), mv, k_type=kt)

    def shuffle(self, img, t, k):
        T = self.T[k]
        before = self.shuf_state(k - 1)
        after = self.shuf_state(k)
        dt = 0.6 / self.clip.fps
        offs, vel = [], []
        for r in range(3):
            def o(tt):
                return before[r] + (after[r] - before[r]) * fx.spring(tt, T + 0.05 * r, 1.45, 0.74)
            offs.append(o(t))
            vel.append((o(t) - o(t - dt)) * (grid_rect(3, 0, 0)[2] + GUT) * 0.9)
        types = {1 - round(before[1]): k - 1, 1 - round(after[1]): k}
        self.g3_static(img, t, k, offs, {"vel": vel, "types": types})

    def hero(self, img, t, k):
        T = self.T[k]
        prev = self.seq[k - 1][0] if k > 0 else "g2"
        e = fx.spring(t, T, 1.5, 0.72)
        dt = 0.6 / self.clip.fps
        # outgoing grid pushed outward
        if e < 0.995:
            push = 1 + 1.7 * fx.expo(t, T, 0.5)
            fade = 1 - fx.smooth(t, T + 0.15, 0.25)
            if prev in ("g3", "shuf"):
                offs = self.shuf_state(k - 1)
                tw = grid_rect(3, 0, 0)[2] + GUT
                for r in range(3):
                    for cc in range(3):
                        if r == 1 and cc == 1:
                            continue
                        v = cc - round(offs[r])
                        x, y, w, h = grid_rect(3, r, cc)
                        cx, cy = x + w / 2 - c.W / 2, y + h / 2 - c.H / 2
                        p = self.tile(w, h, self.pal_g3(r, v), self.zoom_at(t, r * 3 + v % 3))
                        if p is None:
                            continue
                        X, Y = c.W / 2 + cx * push, c.H / 2 + cy * push
                        mv = (cx * 1.7 * 0.18 * (1 - fx.expo(t, T, 0.5)), cy * 1.7 * 0.18 * (1 - fx.expo(t, T, 0.5)))
                        fx.put(img, p, X, Y, scale=1 + 0.4 * (push - 1), opacity=fade, mv=mv)
            else:
                for q in range(4):
                    rr, cc = divmod(q, 2)
                    x, y, w, h = grid_rect(2, rr, cc)
                    cx, cy = x + w / 2 - c.W / 2, y + h / 2 - c.H / 2
                    p = self.tile(w, h, self.pal_g2(q), self.zoom_at(t, self.idx_g2(q)))
                    fx.put(img, p, c.W / 2 + cx * push, c.H / 2 + cy * push, opacity=fade)
        src = grid_rect(3, 1, 1) if prev in ("g3", "shuf") else (c.W / 2 - 150, c.H / 2 - 260, 300, 520)
        R0 = lerp_rect(src, (0, 0, c.W, c.H), e)
        R1 = lerp_rect(src, (0, 0, c.W, c.H), fx.spring(t - dt, T, 1.5, 0.72))
        rad = int(18 * (1 - min(1, e)))
        p = self.tile(R0[2], R0[3], HERO1, 1.02 + 0.03 * fx.smooth(t, T + 0.5, 1.2), natural=True, rad=rad)
        xf = 1 - fx.smooth(e, 0.08, 0.3)
        if xf > 0.01 and p is not None and prev in ("g3", "shuf"):
            tt_ = self.type_tile(R0[2], R0[3], k - 1, max(rad, 1))
            if tt_ is not None:
                c.over(p, tt_, 0, 0, xf)
        if p is not None:
            dz = (R0[2] - R1[2]) / max(R0[2], 1)
            if dz > 0.02:
                p = fx.zoom_blur(p, p.shape[1] / 2, p.shape[0] / 2, min(0.25, dz * 1.6), n=9)
            c.over(img, p, R0[0], R0[1])
        # payoff label: big, black, letters drop in
        b = self.B[k]
        s = b["label"].upper()
        size = int(min(250, c.fit_size(s, "anton", 960)))
        L = b["key_t"] + 0.05
        base_y = 360
        x = c.W / 2 - c.text_w(s, "anton", size) / 2
        f = c.font("anton", size)
        for j, ch in enumerate(s):
            cw = f.getlength(ch)
            if ch != " ":
                ee = fx.spring(t, L + 0.035 * j, 2.0, 0.55)
                if ee > 0.01:
                    tp, base = self.tx(ch, "anton", size, INK)
                    yy = base_y - (1 - ee) * 300
                    c.over(img, self.tx(ch, "anton", size, PAL[0][0])[0], x + 12, yy - base + 12,
                           min(1, ee * 3))
                    yp = base_y - (1 - fx.spring(t - dt, L + 0.035 * j, 2.0, 0.55)) * 300
                    bp, pad = fx.blur_patch(tp, 0, (yy - yp) * 0.9)
                    c.over(img, bp, x - pad, yy - base - pad, min(1, ee * 3))
            x += cw
        sub = (b.get("sub") or "").upper()
        if sub:
            q = fx.spring(t, L + 0.25, 2.0, 0.65)
            sp, sb = self.tx(sub, "mono", 34, (255, 255, 255), 600, 0.2)
            pill = c.fill(c.rrect(sp.shape[1] + 48, 62, 31), INK)
            c.over(pill, sp, 24, 31 - sb + 12)
            fx.put(img, pill, c.W / 2, base_y - size * 0.86 - 52, scale=max(0.01, q), opacity=min(1, q * 2))

    # ---------- frame ----------
    def frame(self, i, rgb, alpha):
        t = i / self.clip.fps
        self.canvas, a = self.fr.place(rgb, alpha)
        self.matte = a
        self.fb = self.fr.face_box(i)
        img = np.empty((c.H, c.W, 3), np.float32)
        img[:] = INK
        k = 0
        for j, T in enumerate(self.T):
            if t >= T:
                k = j
        kind = self.seq[k][0]
        if kind == "g2":
            self.g2(img, t, k)
        elif kind == "g3":
            if t < self.T[k] + 1.0:
                self.g3_split(img, t, k)
            else:
                self.g3_static(img, t, k, self.shuf_state(k))
        elif kind == "shuf":
            self.shuffle(img, t, k)
        else:
            self.hero(img, t, k)
        img = c.vignette(img, 0.18)
        return c.grain(img, i, 2.2)
