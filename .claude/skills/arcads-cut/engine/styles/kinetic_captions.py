"""/kinetic-captions: premium word-by-word kinetic type in the lower third. Every word stands up out of depth on a spring with motion blur, the spoken word glows, keywords slam in huge in electric lime with a chromatic split while the camera punches in with zoom blur, and finished lines tip back and fly up and away. Opens on a light flash and a camera snap as the first word lands."""
import math

import cv2
import numpy as np

import core as c
import fx
from plan import Plan, norm

ACC = (196, 255, 64)       # electric lime, the one accent
WHITE = (255, 255, 255)
BASE_Y = 1500              # vertical centre of the caption lockup
MAXW = 930                 # widest a row may be
REG_Y0, REG_H = 1000, 860  # caption layer region (canvas y)
SMALL = 84                 # small-word size
ROW_GAP = 24


# ---------- local helpers ----------

def warp3d(patch, rx=0.0, ry=0.0, rz=0.0, scale=1.0, z=0.0, persp=1700.0, shade=0.45):
    """Project a patch as a 3D card about its centre. Returns (warped, x0, y0) offsets from the centre, or None."""
    h, w = patch.shape[:2]
    corners = np.float32([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0], [w / 2, h / 2, 0], [-w / 2, h / 2, 0]])
    quad, _ = fx.project(corners, 0, 0, rx, ry, rz, scale, z, persp)
    facing = float((fx.rot3(rx, ry, rz) @ np.float32([0, 0, -1]))[2]) * -1
    if facing < 0.03:
        return None
    x0, y0 = np.floor(quad.min(0)).astype(int) - 2
    x1, y1 = np.ceil(quad.max(0)).astype(int) + 2
    if x1 - x0 < 2 or y1 - y0 < 2 or x1 - x0 > 5000 or y1 - y0 > 3000:
        return None
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, (quad - np.float32([x0, y0])).astype(np.float32))
    out = cv2.warpPerspective(patch, M, (int(x1 - x0), int(y1 - y0)), flags=cv2.INTER_LINEAR, borderValue=0)
    if shade:
        out[..., :3] *= 1 - shade * (1 - facing)
    return out, int(x0), int(y0)


def chroma_rgba(L, px):
    """Chromatic split of a premultiplied layer: red one way, blue the other (fringes keep their own alpha)."""
    if px < 0.6:
        return L
    h, w = L.shape[:2]
    out = L.copy()
    for ch, s in ((0, 1), (2, -1)):
        M = np.float32([[1, 0, px * s], [0, 1, 0]])
        sh = cv2.warpAffine(L[..., [ch, 3]], M, (w, h), borderValue=0)
        out[..., ch] = sh[..., 0]
        out[..., 3] = np.maximum(out[..., 3], sh[..., 1])
    return out


class Style:
    def __init__(self, clip, opts):
        self.clip = clip
        self.P = P = Plan(clip, opts)
        self.fr = c.Framing(clip, 1215)
        self.N = min(clip.n, len(clip.face))
        self.dur = self.N / clip.fps
        self.locks = [self.lockup(b) for b in P.beats]
        fin = [it for it in self.locks[-1]["items"] if it["key"]]
        if fin:
            t_sw = min(fin[-1]["s"] + 0.1, self.dur - 0.36)
            for it in fin:
                it["sweep"] = t_sw
        self.cam = fx.Cam(self.fr, self.camera_keys(), drift=3.0)
        ys = np.arange(c.H, dtype=np.float32)
        u = np.clip((ys - 1080) / (c.H - 1080), 0, 1)
        self.shade_col = (1 - 0.62 * (u * u * (3 - 2 * u)))[:, None, None]  # lower-third darkening for legibility

    # ---- camera: one distinct move per beat ----
    def camera_keys(self):
        B = self.P.beats
        keys = [{"t": 0.0, "zoom": 1.22, "rot": -1.6},
                {"t": 0.0, "zoom": 1.0, "rot": 0.0, "ease": "expo", "dur": 0.55}]
        moves = [
            lambda kt, nx: [{"t": kt, "zoom": 1.14, "ease": "spring", "freq": 2.3, "damp": 0.5},
                            {"t": min(kt + 0.5, nx), "zoom": 1.05, "ease": "smooth", "dur": 0.7}],
            lambda kt, nx: [{"t": kt, "zoom": 1.11, "rot": 2.4, "dx": -24, "ease": "spring", "freq": 1.7, "damp": 0.55},
                            {"t": min(kt + 0.6, nx), "zoom": 1.04, "rot": 0.0, "dx": 0, "ease": "smooth", "dur": 0.9}],
            lambda kt, nx: [{"t": kt, "zoom": 1.15, "dy": -34, "ease": "spring", "freq": 2.5, "damp": 0.45},
                            {"t": min(kt + 0.45, nx), "zoom": 1.07, "dy": 0, "ease": "smooth", "dur": 0.6}],
            lambda kt, nx: [{"t": kt, "zoom": 1.21, "rot": -1.2, "ease": "expo", "dur": 0.16},
                            {"t": min(kt + 0.16, nx), "zoom": 1.05, "rot": 0.0, "ease": "smooth", "dur": 1.1}],
        ]
        for j, b in enumerate(B):
            kt = max(b["key_t"], 0.3 if j == 0 else 0)
            nx = B[j + 1]["key_t"] - 0.05 if j + 1 < len(B) else 99
            keys += moves[j % len(moves)](kt, nx)
        return keys

    # ---- type ----
    def word_item(self, text, s, e, key, size):
        col = ACC if key else WHITE
        if key:
            p, base = c.text_patch(text.upper(), "grotesk", int(size), col, 900, -0.02)
        else:
            p, base = c.text_patch(text, "grotesk", int(size), col, 720, -0.01)
        sig = max(6.0, size * 0.16)
        pad = int(sig * 2.6)
        g = cv2.GaussianBlur(np.pad(p[..., 3] / 255, pad), (0, 0), sig)
        glow = np.concatenate([g[..., None] * np.float32(col), g[..., None] * 255], -1)
        return {"text": text, "s": s, "e": e, "key": key, "patch": p, "base": base, "size": size,
                "w": p.shape[1], "glow": glow, "gpad": pad}

    def lockup(self, b):
        keys = set(b["key"].split())
        ws = [w for w in b["words"] if w["w"].strip(" ,.!?")]
        if not ws:  # no word timing: spread the beat text evenly
            toks = b["text"].split() or [b["label"]]
            d = (b["t1"] - b["t0"]) / len(toks)
            ws = [{"w": tk, "s": b["t0"] + k * d, "e": b["t0"] + (k + 1) * d} for k, tk in enumerate(toks)]
        runs = []
        for w in ws:
            s = w["w"].strip(" ,.!?")
            k = norm(s) in keys
            if runs and runs[-1][0] == k:
                runs[-1][1].append((s, w["s"], w["e"]))
            else:
                runs.append([k, [(s, w["s"], w["e"])]])
        rows = []
        for k, items in runs:
            if k:
                joined = " ".join(x[0] for x in items).upper()
                size = min(210, c.fit_size(joined, "grotesk", MAXW, 900, -0.02))
                if size >= 150 or len(items) == 1:
                    rows.append({"key": True, "size": size, "words": items})
                else:
                    for x in items:
                        rows.append({"key": True, "size": min(210, c.fit_size(x[0].upper(), "grotesk", MAXW, 900,
                                                                           -0.02)), "words": [x]})
            else:
                cur, cw = [], 0
                for x in items:
                    ww = c.text_w(x[0], "grotesk", SMALL, 720, -0.01) + SMALL * 0.26
                    if cur and cw + ww > MAXW:
                        rows.append({"key": False, "size": SMALL, "words": cur})
                        cur, cw = [], 0
                    cur.append(x)
                    cw += ww
                if cur:
                    rows.append({"key": False, "size": SMALL, "words": cur})
        # fit the lockup height
        def heights(rs, k=1.0):
            return [(r["size"] * k * 0.74, (0 if r["key"] else r["size"] * k * 0.24)) for r in rs]
        k = 1.0
        while True:
            hs = heights(rows, k)
            tot = sum(a + d for a, d in hs) + ROW_GAP * (len(rows) - 1)
            if tot <= 560 or k < 0.6:
                break
            k *= 0.94
        items = []
        y = BASE_Y - REG_Y0 - tot / 2
        for r, (asc, desc) in zip(rows, hs):
            size = r["size"] * k
            y += asc
            its = [self.word_item(x[0], x[1], x[2], r["key"], size) for x in r["words"]]
            gap = size * (0.22 if r["key"] else 0.26)
            rw = sum(it["w"] for it in its) + gap * (len(its) - 1)
            x = 540 - rw / 2
            arr = []
            for it in its:
                it["cx"] = x + it["w"] / 2
                it["cy"] = y - it["base"] + it["patch"].shape[0] / 2
                x += it["w"] + gap
                arr.append((it["s"] - 0.05, (rw - (x - gap - (540 - rw / 2))) / 2))
                it["row"] = arr
                items.append(it)
            y += desc + ROW_GAP
        return {"items": items, "t0": b["t0"], "top": BASE_Y - tot / 2, "bot": BASE_Y + tot / 2}

    # ---- caption layer ----
    @staticmethod
    def row_shift(arr, t):
        """Rows stay centred on the words already spoken: each new word springs the row sideways."""
        sh = arr[0][1]
        for k in range(1, len(arr)):
            sh += (arr[k][1] - arr[k - 1][1]) * fx.spring(t, arr[k][0], 2.4, 0.78)
        return sh

    def draw_lockup(self, lk, t, L, first=False):
        items = lk["items"]
        fps = self.clip.fps
        for n, it in enumerate(items):
            ts = it["s"] - 0.05
            if t < ts:
                continue
            nxt = items[n + 1]["s"] if n + 1 < len(items) else it["e"] + 0.25
            active = it["s"] - 0.02 <= t < max(nxt, it["e"]) + 0.02
            xs = self.row_shift(it["row"], t)
            if it["key"] or (first and n == 0):
                self.draw_key(it, t, ts, L, active, xs, opener=not it["key"])
                continue
            st = lambda tt: (fx.spring(tt, ts, 2.4, 0.52))
            e, ep = st(t), st(t - 0.6 / fps)
            rx = -82 * (1 - e)
            dy, dyp = 74 * (1 - e), 74 * (1 - ep)
            op = c.clamp((t - ts) / 0.09) * (1.0 if active or t < it["s"] + 0.3 else 0.8)
            r = warp3d(it["patch"], rx=rx, z=260 * (1 - e))
            if r is None:
                continue
            p, x0, y0 = r
            p, pad = fx.blur_patch(p, 0, dy - dyp)
            c.over(L, p, it["cx"] + xs + x0 - pad, it["cy"] + dy + y0 - pad, op)
            if active:
                g = 0.55 * c.clamp((t - it["s"] + 0.02) / 0.08)
                gp = it["glow"]
                fx.add(L, gp, it["cx"] + xs - gp.shape[1] / 2, it["cy"] + dy - gp.shape[0] / 2, g)

    def draw_key(self, it, t, ts, L, active, xs=0.0, opener=False):
        fps = self.clip.fps
        s0 = 2.3 if opener else 1.9
        sc = lambda tt: c.lerp(s0, 1.0, fx.spring(tt, ts, 3.1, 0.52)) * (1 + 0.05 * fx.smooth(tt, ts + 0.25, 1.2))
        s, sp = sc(t), sc(t - 0.6 / fps)
        ry = (0 if opener else 30) * (1 - fx.spring(t, ts, 2.6, 0.55))
        op = c.clamp((t - ts) / 0.05) if not opener else 1.0
        slam = t - ts < 0.4
        K = c.layer(REG_H, c.W) if slam else L
        patch = it["patch"]
        if it.get("sweep"):  # the payoff: one specular sweep across the last keyword once it has settled
            patch = fx.light_sweep(patch, c.prog(t, it["sweep"], 0.4), width=0.16, strength=0.85)
        r = warp3d(patch, ry=ry, scale=s, shade=0.3)
        if r is not None:
            p, x0, y0 = r
            c.over(K, p, it["cx"] + xs + x0, it["cy"] + y0, op)
        # accent glow: a hit on landing, then breathing while spoken
        hit = math.exp(-max(t - ts - 0.05, 0) / 0.18)
        g = (0.55 * hit + (0.32 if active else 0.12)) if not opener else (0.45 * hit + (0.5 if active else 0))
        gp = it["glow"]
        if s != 1.0:
            gp = cv2.resize(gp, (max(1, int(gp.shape[1] * s)), max(1, int(gp.shape[0] * s))))
        fx.add(K, gp, it["cx"] + xs - gp.shape[1] / 2, it["cy"] - gp.shape[0] / 2, g * op)
        if slam:
            dz = (sp - s) / max(s, 1e-3)
            if abs(dz) > 0.012:
                K = fx.zoom_blur(K, it["cx"] + xs, it["cy"], min(dz * 1.4, 0.35), n=7)
            K = chroma_rgba(K, 24 * math.exp(-max(t - ts, 0) / 0.11))
            c.over(L, K)

    # ---- frame ----
    def frame(self, i, rgb, alpha):
        t = i / self.clip.fps
        img, _ = self.cam.place(i, t, rgb, alpha)
        img = c.grade(img, contrast=1.1, sat=1.02, warm=0.006, gain=0.97)
        img *= self.shade_col
        # captions: current lockup, plus the previous one tipping back and flying away
        B = self.P.beats
        j = max([k for k, b in enumerate(B) if b["t0"] - 0.06 <= t] or [0])
        for k in ((j - 1, j) if j else (j,)):
            lk = self.locks[k]
            L = c.layer(REG_H, c.W)
            self.draw_lockup(lk, t, L, first=k == 0)
            if k < j:  # exit
                tx = B[j]["t0"] - 0.06
                ex = lambda tt: fx.expo(tt, tx, 0.42)
                p, pp = ex(t), ex(t - 0.6 / self.clip.fps)
                if p >= 0.999:
                    continue
                dy, dyp = -300 * p, -300 * pp
                sh, (ox, oy) = c.shadow(L, 14, 0.55, 6)
                c.over(sh, L, -ox, -oy)
                Lb, pad = fx.blur_patch(sh, 0, (dy - dyp) * 1.2)
                r = warp3d(Lb, rx=58 * p, scale=1 - 0.12 * p, shade=0.4)
                if r is not None:
                    q, x0, y0 = r
                    c.over(img, q, 540 + x0, REG_Y0 + REG_H / 2 + dy + y0 + oy, 1 - fx.smooth(t, tx, 0.34))
            else:
                sh, (ox, oy) = c.shadow(L, 14, 0.55, 6)
                c.over(img, sh, ox, REG_Y0 + oy)
                c.over(img, L, 0, REG_Y0)
        # opening light flash (the first word lands into it)
        fl = 1 - fx.expo(t, 0.0, 0.32)
        if fl > 0.01:
            img = img + (255 - img) * (fl * 0.88)
        img = c.bloom(img, 200, 26, 0.32)
        img = c.vignette(img, 0.3)
        return c.grain(img, i, 2.2)
