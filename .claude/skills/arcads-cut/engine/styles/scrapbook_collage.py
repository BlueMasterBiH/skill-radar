"""/scrapbook-collage: tactile stop-motion zine. A red cover sheet rips open down a torn edge to reveal the speaker as a die-cut sticker on kraft paper; each beat's label flies in as ransom-note letter cards that flip in 3D and land with a bounce, tape slaps on, a label-maker strip, a typed paper strip, a film strip of live frames and a rubber stamp take turns; all paper boils on twos (12 fps) while the speaker stays smooth at 25 fps."""
import math
import zlib

import cv2
import numpy as np

import core as c
import fx
from plan import Plan

KRAFT = (192, 156, 112)
INK = (24, 22, 20)
RED = (214, 44, 36)
YEL = (255, 210, 0)
CARDS = [((250, 248, 240), INK), ((22, 22, 22), (250, 248, 240)), (RED, (255, 255, 255)), (YEL, INK),
         ((232, 225, 206), INK), ((64, 104, 196), (255, 255, 255)), ((248, 188, 198), RED)]
FONTS = [("anton", None), ("archivo", None), ("display", 900), ("serif", None), ("marker", None),
         ("spacemono", None), ("montserrat", 800), ("grotesk", 900), ("display-i", 700)]
SHUT = 0.5


def smoothstep(x, a, b):
    y = np.clip((x - a) / (b - a), 0, 1)
    return y * y * (3 - 2 * y)


def torn(w, h, seed, amp=10, edges="tblr", fine=2.0):
    """Torn-paper alpha (h, w): ragged edges with fine fibre noise."""
    rng = np.random.default_rng(seed)
    m = np.ones((h, w), np.float32)
    ys, xs = np.mgrid[0:h, 0:w]
    for e in edges:
        L = w if e in "tb" else h
        prof = np.cumsum(rng.normal(0, 1.4, L))
        prof = (prof - prof.min()) / (np.ptp(prof) + 1e-6) * amp + rng.normal(0, fine, L) + 2
        if e == "t":
            m *= np.clip(ys - prof[None, :], 0, 1)
        elif e == "b":
            m *= np.clip((h - 1 - ys) - prof[None, :], 0, 1)
        elif e == "l":
            m *= np.clip(xs - prof[:, None], 0, 1)
        else:
            m *= np.clip((w - 1 - xs) - prof[:, None], 0, 1)
    return m.astype(np.float32)


def paper_tex(h, w, color, seed, fib=6.0, grain=4.0):
    rng = np.random.default_rng(seed)
    base = np.empty((h, w, 3), np.float32)
    base[:] = color
    n1 = cv2.resize(cv2.GaussianBlur(rng.normal(0, 1, (h // 6 + 1, w // 6 + 1)).astype(np.float32), (0, 0), 4), (w, h))
    n2 = cv2.blur(rng.normal(0, 1, (h, w)).astype(np.float32), (25, 1))
    n3 = rng.normal(0, 1, (h, w)).astype(np.float32)
    v = n1 * fib * 1.4 + n2 * fib + n3 * grain
    return np.clip(base + v[..., None], 0, 255)


def card_warp(dst, patch, cx, cy, rx=0.0, ry=0.0, rz=0.0, s=1.0, z=0.0, back=None, lift=0.0, op=1.0, persp=1500.0):
    """A paper card in 3D with a small paper shadow that grows as it lifts off the page."""
    h, w = patch.shape[:2]
    pts = np.float32([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0], [w / 2, h / 2, 0], [-w / 2, h / 2, 0]])
    quad, zz = fx.project(pts, cx, cy, rx, ry, rz, s, z, persp)
    if (zz < 80).any():
        return
    nrm = fx.rot3(rx, ry, rz) @ np.float32([0, 0, -1])
    facing = float(-nrm[2])
    src = patch
    if facing < 0:
        src = back[:, ::-1].copy() if back is not None else patch[:, ::-1].copy()
    x0, y0 = np.floor(quad.min(0)).astype(int) - 2
    x1, y1 = np.ceil(quad.max(0)).astype(int) + 2
    if x1 - x0 < 2 or y1 - y0 < 2 or x1 < 0 or y1 < 0 or x0 > c.W or y0 > c.H:
        return
    sq = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(sq, (quad - np.float32([x0, y0])).astype(np.float32))
    out = cv2.warpPerspective(src, M, (int(x1 - x0), int(y1 - y0)), flags=cv2.INTER_LINEAR, borderValue=0)
    out[..., :3] *= 1 - 0.3 * (1 - abs(facing))
    sb = 4 + lift * 16
    sh, (ox, oy) = c.shadow(out, int(sb), 0.42 - lift * 0.15, int(4 + lift * 30))
    c.over(dst, sh, x0 + ox + 3 + lift * 14, y0 + oy, op)
    c.over(dst, out, x0, y0, op)


class Style:
    def __init__(self, clip, opts):
        self.clip = clip
        self.P = P = Plan(clip, opts)
        self.fr = c.Framing(clip, 1215)
        self.B = B = P.beats
        self.n = n = len(B)
        self.N = min(clip.n, len(clip.face))
        self.dur = self.N / clip.fps
        self.kraft = paper_tex(c.H, c.W, KRAFT, 1, 7.0, 5.0)
        # big torn black paper behind the speaker, a newsprint scrap behind the letters
        bw, bh = 940, 1240
        blk = paper_tex(bh, bw, (28, 26, 25), 2, 3.0, 2.5)
        m = torn(bw, bh, 3, 16, "tblr")
        self.black = np.concatenate([blk * m[..., None], m[..., None] * 255], -1)
        nw, nh = 700, 330
        news = paper_tex(nh, nw, (234, 228, 212), 4, 3.0, 3.0)
        rng = np.random.default_rng(6)
        for col in range(3):
            for row in range(9):
                L = int(rng.uniform(140, 200))
                y = 40 + row * 30
                x = 30 + col * 220
                news[y:y + 9, x:x + L] = news[y:y + 9, x:x + L] * 0.55
        m = torn(nw, nh, 7, 12, "tblr")
        self.news = np.concatenate([news * m[..., None], m[..., None] * 255], -1)
        # red cover that tears open
        self.cover = paper_tex(c.H, c.W, (206, 40, 34), 9, 6.0, 4.0)
        self.tear_x = self._tear_line()
        self.letters = [self._ransom(k) for k in range(n)]
        self._tx = {}
        self.tapes = {}

    # ---------- helpers ----------
    def tg(self, i):
        """Stop-motion clock: graphics move on twos."""
        return (i // 2) * 2 / self.clip.fps

    def boil(self, key, i, amp=1.6, rot=0.7):
        rng = np.random.default_rng(zlib.crc32(repr((key, i // 2)).encode()))
        return rng.uniform(-amp, amp), rng.uniform(-amp, amp), rng.uniform(-rot, rot)

    def tx(self, s, key, size, color, wght=None, tracking=0.0):
        kk = (s, key, size, color, wght, tracking)
        if kk not in self._tx:
            self._tx[kk] = c.text_patch(s, key, size, color, wght, tracking)
        return self._tx[kk]

    def tape(self, w, h, seed):
        kk = (w, h, seed)
        if kk not in self.tapes:
            m = torn(w, h, seed, 4, "lr", 1.5)
            col = paper_tex(h, w, (238, 228, 196), seed, 2.0, 2.0)
            col[:, :, :] += np.linspace(14, -6, h, dtype=np.float32)[:, None, None]
            self.tapes[kk] = np.concatenate([col * m[..., None] * 0.8, m[..., None] * 255 * 0.8], -1)
        return self.tapes[kk]

    def _tear_line(self):
        rng = np.random.default_rng(12)
        ys = np.arange(c.H)
        drift = np.cumsum(rng.normal(0, 1.0, c.H))
        drift = cv2.GaussianBlur(drift.reshape(-1, 1).astype(np.float32), (0, 0), 18).ravel() * 3.2
        return 560 - ys * 0.05 + drift + rng.normal(0, 2.2, c.H)

    def _ransom(self, k):
        b = self.B[k]
        text = b["label"].upper()
        words = text.split()
        lines = [text]
        if len(text) > 9 and len(words) > 1:
            best = min(range(1, len(words)), key=lambda m: abs(len(" ".join(words[:m])) - len(" ".join(words[m:]))))
            lines = [" ".join(words[:best]), " ".join(words[best:])]
        rng = np.random.default_rng(40 + k)
        size = 172 if len(lines) == 1 else 124
        out = []
        for li, ln in enumerate(lines):
            cards = []
            for ch in ln:
                if ch == " ":
                    cards.append(None)
                    continue
                fk, wg = FONTS[int(rng.integers(0, len(FONTS)))]
                bg, ink = CARDS[int(rng.integers(0, len(CARDS)))]
                s = int(size * rng.uniform(0.8, 1.1))
                g = ch if rng.random() < 0.82 or fk in ("anton", "archivo") else ch.lower()
                tp, base = c.text_patch(g, fk, s, ink, wg, 0.0, 4)
                ys = np.nonzero(tp[..., 3].max(1) > 8)[0]
                xs = np.nonzero(tp[..., 3].max(0) > 8)[0]
                tp = tp[ys[0]:ys[-1] + 1, xs[0]:xs[-1] + 1]
                px, py = int(rng.uniform(10, 22)), int(rng.uniform(10, 20))
                w, h = tp.shape[1] + 2 * px, tp.shape[0] + 2 * py
                m = torn(w, h, int(rng.integers(0, 1e6)), 3, "tblr", 0.8)
                face = paper_tex(h, w, bg, int(rng.integers(0, 1e6)), 2.0, 2.0)
                cardp = np.concatenate([face * m[..., None], m[..., None] * 255], -1)
                c.over(cardp, tp, px, py)
                backp = np.concatenate([paper_tex(h, w, (214, 204, 182), 5, 2.0, 2.0) * m[..., None],
                                        m[..., None] * 255], -1)
                cards.append({"p": cardp, "back": backp, "rot": rng.uniform(-9, 9), "dy": rng.uniform(-10, 10),
                              "from": (rng.uniform(-500, 500), rng.uniform(-700, -300)), "flip": rng.choice(["x", "y"])})
            widths = [(cd["p"].shape[1] - 6) if cd else size * 0.3 for cd in cards]
            tot = sum(widths)
            sc = min(1.0, 990 / tot)
            x = 540 - tot * sc / 2
            y = 250 + li * size * 1.22 if len(lines) > 1 else 300
            for cd, wd in zip(cards, widths):
                if cd:
                    cd["x"], cd["y"], cd["s"] = x + wd * sc / 2, y + cd["dy"], sc
                    out.append(cd)
                x += wd * sc
        return out

    # ---------- pieces ----------
    def sticker(self, img, i, t, canvas, a):
        tg = self.tg(i)
        ah = smoothstep(a, 0.38, 0.62)
        hard = cv2.morphologyEx((ah > 0.5).astype(np.uint8), cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
        nl, lab, stats, _ = cv2.connectedComponentsWithStats(hard, connectivity=8)
        if nl > 2:  # drop matte islands (desk, laptop) so the die-cut stays one clean shape
            keep = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            hard = (lab == keep).astype(np.uint8)
            ah = ah * hard
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
        ring = cv2.GaussianBlur(cv2.dilate(hard, k).astype(np.float32), (0, 0), 1.3)
        person = c.grade(canvas, contrast=1.08, sat=1.06)
        rgbo = person * ah[..., None] + 250 * (1 - ah[..., None])
        st = np.concatenate([rgbo * ring[..., None], ring[..., None] * 255], -1)
        # slaps down after the tear: big and lifted, then pressed flat
        e = fx.spring(tg, 0.16, 2.2, 0.5)
        s = 0.86 * (1.14 - 0.14 * e) if tg >= 0.16 else 0.86 * 1.14
        bx, by, br = self.boil("st", i, 1.2, 0.35)
        rot = -2.5 + br
        M = cv2.getRotationMatrix2D((540, 1920), rot, s)
        M[0, 2] += bx
        M[1, 2] += by + 30
        warped = cv2.warpAffine(st, M, (c.W, c.H), flags=cv2.INTER_LINEAR, borderValue=0)
        lift = 1 - e if tg >= 0.16 else 1.0
        sm = cv2.resize(warped[..., 3], (c.W // 4, c.H // 4), interpolation=cv2.INTER_AREA)
        sh = cv2.resize(cv2.GaussianBlur(sm, (0, 0), 3 + 6 * lift), (c.W, c.H)) / 255
        sh = np.roll(np.roll(sh, int(16 + 30 * lift), 0), int(10 + 16 * lift), 1)
        img *= (1 - sh[..., None] * (0.5 - 0.15 * lift))
        c.over(img, warped)
        return M

    def ransom(self, img, i, k):
        tg = self.tg(i)
        for j in (k - 1, k):
            if j < 0:
                continue
            L = self.B[j]["key_t"]
            t_out = self.B[j + 1]["key_t"] - 0.22 if j + 1 < self.n else 99
            for q, cd in enumerate(self.letters[j]):
                ts = L - 0.12 + 0.04 * q
                if tg < ts:
                    continue
                e = fx.spring(tg, ts, 1.9, 0.42)
                ep = fx.spring(tg, ts, 1.6, 0.75)
                fx_, fy_ = cd["from"]
                x = cd["x"] + fx_ * (1 - ep)
                y = cd["y"] + fy_ * (1 - ep)
                flip = 180 * (1 - e)
                rx, ry = (flip, 0.0) if cd["flip"] == "x" else (0.0, flip)
                z = -420 * (1 - ep)
                lift = c.clamp(1 - ep)
                op = 1.0
                if tg >= t_out:
                    qq = fx.smooth(tg, t_out + 0.02 * q, 0.3)
                    x += (q - len(self.letters[j]) / 2) * 60 * qq
                    y -= 900 * qq * qq
                    rx += 120 * qq
                    lift = max(lift, qq)
                    z -= 200 * qq
                    if qq >= 1:
                        continue
                bx, by, br = self.boil(("L", j, q), i, 1.5, 0.8)
                card_warp(img, cd["p"], x + bx, y + by, rx, ry, cd["rot"] + br, cd["s"], z, cd["back"], lift, op)

    def tapes_on(self, img, i, t):
        """Tape slaps on the black paper corners on the first keyword."""
        tg = self.tg(i)
        L = self.B[0]["key_t"]
        spots = [(140, 330, -38, 200, 0.0), (950, 300, 34, 190, 0.06), (930, 1500, -30, 180, 0.12)]
        for n_, (x, y, rot, w, d) in enumerate(spots):
            ts = L + 0.05 + d
            if tg < ts:
                continue
            e = fx.spring(tg, ts, 2.6, 0.5)
            s = 1.35 - 0.35 * e
            bx, by, br = self.boil(("T", n_), i, 1.2, 0.5)
            p = self.tape(w, 58, 30 + n_)
            fx.put(img, p, x + bx, y + by, scale=s, rot=rot + br, opacity=min(1, e * 3),
                   shadow=0.18, sblur=4, sdy=3)

    def sub_piece(self, img, i, t, k, canvas):
        """A different paper piece carries each beat's sub line."""
        tg = self.tg(i)
        n = self.n
        for j in (k - 1, k):
            if j < 0:
                continue
            b = self.B[j]
            sub = (b.get("sub") or b["label"]).upper()
            L = b["key_t"] + 0.22
            if tg < L:
                continue
            t_out = self.B[j + 1]["key_t"] - 0.22 if j + 1 < n else 99
            out = fx.smooth(tg, t_out, 0.3) if tg >= t_out else 0.0
            if out >= 1:
                continue
            kind = ["dymo", "typed", "film", "dymo"][j % 4] if j < n - 1 or n == 1 else "stamp"
            bx, by, br = self.boil(("S", j), i, 1.4, 0.6)
            ylab = 520 if len(self.letters[j]) and max(cd["y"] for cd in self.letters[j]) > 330 else 470
            if kind == "dymo":
                tp, base = self.tx(sub, "spacemono", 40, (250, 250, 250), None, 0.16)
                w, h = tp.shape[1] + 60, 76
                p = c.fill(c.rrect(w, h, 10), (16, 16, 18))
                p[6:24, 8:-8, :3] += 22 * p[6:24, 8:-8, 3:4] / 255
                gl = cv2.GaussianBlur(tp, (0, 0), 1.2)
                c.over(p, gl * 0.5, 31, h / 2 - base + 14)
                c.over(p, tp, 30, h / 2 - base + 13)
                e = fx.spring(tg, L, 2.4, 0.5)
                fx.put(img, p, 540 + bx, ylab + by + 110 * out * out - 0 * e, scale=1.3 - 0.3 * e, rot=-3 + br + 40 * out,
                       opacity=min(1, e * 3) * (1 - out), shadow=0.3, sblur=5, sdy=4)
            elif kind == "typed":
                n_ch = int(max(0, tg - L) * 34)
                full, base = self.tx(sub, "spacemono", 38, INK, None, 0.08)
                w, h = full.shape[1] + 70, 92
                if (j, "typed") not in self.tapes:
                    m = torn(w, h, 77 + j, 6, "lr", 1.2)
                    self.tapes[(j, "typed")] = np.concatenate([paper_tex(h, w, (250, 248, 240), 8, 2, 2) * m[..., None],
                                                               m[..., None] * 255], -1)
                p = self.tapes[(j, "typed")].copy()
                if n_ch > 0:
                    tp, base = self.tx(sub[:n_ch], "spacemono", 38, INK, None, 0.08)
                    c.over(p, tp, 35, h / 2 - base + 14)
                e = fx.spring(tg, L - 0.08, 2.0, 0.6)
                x = 540 - (1 - e) * 900 + bx
                fx.put(img, p, x, ylab + 46 + by - 700 * out * out, rot=2 + br, opacity=1 - out, shadow=0.3, sblur=5, sdy=4)
                # red marker swoosh under the label
                pr = c.prog(tg, L + 0.06, 0.32)
                if pr > 0:
                    lt = self.letters[j]
                    x0 = min(cd["x"] for cd in lt) - 30
                    x1 = max(cd["x"] for cd in lt) + 30
                    yb = max(cd["y"] for cd in lt) + 78
                    xs = np.linspace(x0, x1, 40)
                    ys = yb + 10 * np.sin(np.linspace(0, 3.1, 40)) - np.linspace(0, 18, 40)
                    c.polyline(img, np.stack([xs, ys], 1), RED, 11, pr, 0.92 * (1 - out))
            elif kind == "film":
                fw, fh = 210, 150
                strip = c.fill(np.ones((fh + 76, 3 * fw + 4 * 18), np.float32), (18, 16, 16))
                for f in range(3):
                    zoom = [1.0, 1.35, 1.8][f]
                    fb = self.fr.face_box(min(i, self.N - 1))
                    cw_, ch_ = fw * 2.6 / zoom, fh * 2.6 / zoom
                    x0 = int(c.clamp(fb[0] - cw_ / 2, 0, c.W - cw_))
                    y0 = int(c.clamp(fb[1] - ch_ * 0.45, 0, c.H - ch_))
                    crop = cv2.resize(canvas[y0:y0 + int(ch_), x0:x0 + int(cw_)], (fw, fh), interpolation=cv2.INTER_AREA)
                    crop = c.grade(crop, contrast=1.15, sat=0.8, warm=0.03)
                    strip[38:38 + fh, 18 + f * (fw + 18):18 + f * (fw + 18) + fw, :3] = crop
                for hx in range(14, strip.shape[1] - 14, 34):
                    for hy in (12, fh + 50):
                        strip[hy:hy + 14, hx:hx + 18, :3] = (232, 226, 210)
                tp, base = self.tx(sub, "spacemono", 30, YEL, None, 0.18)
                lab = c.fill(np.ones((54, tp.shape[1] + 30), np.float32), INK)
                c.over(lab, tp, 15, 27 - base + 10)
                e = fx.spring(tg, L - 0.1, 1.7, 0.6)
                x = 360 - (1 - e) * 800 + bx
                fx.put(img, strip, x, 1540 + by + 600 * out * out, rot=7 + br - 10 * (1 - e), opacity=1 - out,
                       shadow=0.4, sblur=8, sdy=6)
                fx.put(img, lab, x + 10, 1540 - (fh + 76) / 2 - 30 + by + 600 * out * out, rot=7 + br,
                       opacity=(1 - out) * min(1, e * 2), shadow=0.25, sblur=4, sdy=3)
                tp2 = self.tape(150, 50, 91)
                fx.put(img, tp2, x - 300, 1500 + by, rot=-50, opacity=min(1, e * 2) * (1 - out))
            else:  # stamp
                tp, base = self.tx(sub, "archivo", 64, RED, None, 0.06)
                w, h = tp.shape[1] + 80, 132
                p = np.zeros((h, w, 4), np.float32)
                ring = c.rrect(w, h, 18) - np.pad(c.rrect(w - 16, h - 16, 12), 8)
                ring2 = np.pad(c.rrect(w - 28, h - 28, 9) - np.pad(c.rrect(w - 34, h - 34, 7), 3), 14)
                c.over(p, c.fill(np.clip(ring + ring2, 0, 1), RED))
                c.over(p, tp, 40, h / 2 - base + 22)
                rng = np.random.default_rng(55)
                ink = np.clip(cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 1.6) * 2.4 - 0.25, 0, 1)
                p *= (0.25 + 0.75 * ink)[..., None]
                e = fx.spring(tg, L - 0.12, 2.6, 0.45)
                fx.put(img, p, 560 + bx * 0.5, ylab + 66 + by * 0.5, scale=1.9 - 0.9 * e, rot=-7,
                       opacity=min(1, e * 2.5) * 0.92)

    def doodles(self, img, i, t):
        """Yellow marker stars pop around the head on the payoff."""
        if self.n < 2:
            return
        tg = self.tg(i)
        L = self.B[-1]["key_t"] + 0.3
        stars = [(170, 640, 46, 0.0), (900, 560, 60, 0.08), (860, 820, 34, 0.16)]
        for n_, (x, y, r, d) in enumerate(stars):
            pr = c.prog(tg, L + d, 0.24)
            if pr <= 0:
                continue
            bx, by, br = self.boil(("D", n_), i, 2.0, 0)
            starp = [(x + bx + r * math.cos(-math.pi / 2 + m * 4 * math.pi / 5), y + by + r * math.sin(-math.pi / 2 + m * 4 * math.pi / 5))
                     for m in range(6)]
            c.polyline(img, starp, YEL, 9, pr)

    def tear(self, img, i, t):
        """The red cover rips open down a ragged edge; the halves fly apart on twos."""
        tg = self.tg(i)
        x = c.clamp((tg - 0.04) / 0.3)
        if x >= 1:
            return img
        e = x ** 1.5
        ys = np.arange(c.H, dtype=np.float32)
        xs = np.arange(c.W, dtype=np.float32)
        line = self.tear_x[:, None]
        rng = np.random.default_rng(3)
        if not hasattr(self, "_halves"):
            fib = cv2.resize(rng.random((c.H // 2, 8)).astype(np.float32), (1, c.H))[:, 0] * 10 + 5
            left = (xs[None, :] < line).astype(np.float32)
            right = 1 - left
            dist = np.abs(xs[None, :] - line)
            core = (dist < fib[:, None]).astype(np.float32)
            core = cv2.GaussianBlur(core, (0, 0), 0.8)
            col = self.cover.copy()
            # title in big ransom-style type on the cover
            title = (self.P.title or "").upper()
            if title:
                f = int(min(170, c.fit_size(title, "anton", 900)))
                tp, base = c.text_patch(title, "anton", f, (250, 246, 236))
                cc = np.concatenate([col, np.full((c.H, c.W, 1), 255, np.float32)], -1)
                c.over(cc, tp, 540 - tp.shape[1] / 2, 820 - base)
                col = cc[..., :3]
            col = col * (1 - core[..., None]) + np.float32([246, 240, 228]) * core[..., None]
            self._halves = []
            for m in (left, right):
                self._halves.append(np.concatenate([col * m[..., None], m[..., None] * 255], -1))
        for side, half in zip((-1, 1), self._halves):
            dx = side * 820 * e
            dy = -160 * e
            rot = side * 22 * e
            M = cv2.getRotationMatrix2D((540 + side * 540, 1920 if side < 0 else 0), rot, 1.0)
            M[0, 2] += dx
            M[1, 2] += dy
            xp = c.clamp((tg - 0.04 - 0.08) / 0.3)
            pv = side * 820 * max(0, xp) ** 1.5
            mv = (dx - pv) * 0.6
            out = cv2.warpAffine(half, M, (c.W, c.H), flags=cv2.INTER_LINEAR, borderValue=0)
            if abs(mv) > 2:
                out = fx.dir_blur(out, mv, 0)
            sh, (ox, oy) = c.shadow(out[::4, ::4], 4, 0.4, 3)
            c.over(img, cv2.resize(sh, (sh.shape[1] * 4, sh.shape[0] * 4)), ox * 4 + side * 8, oy * 4)
            c.over(img, out)
        return img

    # ---------- frame ----------
    def frame(self, i, rgb, alpha):
        t = i / self.clip.fps
        tg = self.tg(i)
        canvas, a = self.fr.place(rgb, alpha)
        img = self.kraft.copy()
        k = 0
        for j, b in enumerate(self.B):
            if tg >= b["key_t"] - 0.22:
                k = j
        bx, by, br = self.boil("news", i, 1.0, 0.4)
        fx.put(img, self.news, 760 + bx, 300 + by, rot=5 + br, shadow=0.35, sblur=8, sdy=6)
        bx, by, br = self.boil("black", i, 1.0, 0.3)
        fx.put(img, self.black, 540 + bx, 1130 + by, rot=-3.5 + br, shadow=0.45, sblur=12, sdy=10)
        self.sticker(img, i, t, canvas, a)
        self.tapes_on(img, i, t)
        self.sub_piece(img, i, t, k, canvas)
        self.ransom(img, i, k)
        self.doodles(img, i, t)
        img = self.tear(img, i, t)
        img = c.vignette(img, 0.3)
        return c.grain(img, i // 2, 3.0)
