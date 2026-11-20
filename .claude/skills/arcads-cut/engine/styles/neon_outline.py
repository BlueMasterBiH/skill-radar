"""/neon-outline: a dark, moody night look where the wall drops away and only the speaker stays lit. Two electric neon tubes race up both sides of his silhouette and meet over his head with a flicker-on, then trace him for the whole clip; each idea buzzes on as a flickering neon sign on the wall behind him (he stands in front of it, its glow spilling on the wall and wrapping his hair), the tube changes colour with an energy pulse, a neon shockwave echoes off him on the big word and the payoff surges to a double tube."""
import math

import cv2
import numpy as np

import core as c
import fx
from plan import Plan

PINK = (255, 44, 170)
CYAN = (30, 224, 255)
AMBER = (255, 170, 40)
VIOLET = (160, 92, 255)
SIGN_COLS = [CYAN, PINK, AMBER, VIOLET]
HW, HH = c.W // 2, c.H // 2


# ---------- path helpers ----------

def silhouette(a):
    """Open outline path of the person (canvas px), from one frame edge round the head to the other."""
    sm = cv2.resize(a, (HW, HH), interpolation=cv2.INTER_AREA)
    sm = cv2.GaussianBlur(sm, (0, 0), 5)
    m = (sm > 0.3).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return None
    pts = max(cs, key=len)[:, 0].astype(np.float32)
    if len(pts) < 40:
        return None
    border = (pts[:, 0] <= 1) | (pts[:, 0] >= HW - 2) | (pts[:, 1] >= HH - 2) | (pts[:, 1] <= 1)
    n = len(pts)
    if border.all():
        return None
    if border.any():
        b = np.r_[border, border].astype(np.int8)
        d = np.diff(np.r_[1, b, 1])
        starts, ends = np.where(d == -1)[0], np.where(d == 1)[0]
        lens = ends - starts
        k = int(np.argmax(lens))
        s0, L = int(starts[k]), int(min(lens[k], n))
        path = pts[np.arange(s0, s0 + L) % n]
    else:
        k = int(np.argmax(pts[:, 1]))
        path = np.roll(pts, -k, 0)
    if len(path) > 12:
        r = 4
        ker = np.ones(2 * r + 1, np.float32) / (2 * r + 1)
        path = np.stack([np.convolve(np.r_[[path[0, j]] * r, path[:, j], [path[-1, j]] * r], ker, "valid")
                         for j in range(2)], 1)
    return path * 2


def sub(path, a, b):
    """Part of a polyline between arc-length fractions a..b."""
    if path is None or len(path) < 2 or b <= a:
        return None
    seg = np.hypot(*np.diff(path, axis=0).T)
    cum = np.r_[0, np.cumsum(seg)]
    L = cum[-1]
    if L < 1:
        return None
    s0, s1 = a * L, b * L
    inner = path[(cum > s0) & (cum < s1)]
    p0 = np.array([np.interp(s0, cum, path[:, 0]), np.interp(s0, cum, path[:, 1])], np.float32)
    p1 = np.array([np.interp(s1, cum, path[:, 0]), np.interp(s1, cum, path[:, 1])], np.float32)
    return np.vstack([p0, inner, p1])


def tip(path, p):
    seg = np.hypot(*np.diff(path, axis=0).T)
    cum = np.r_[0, np.cumsum(seg)]
    s = p * cum[-1]
    return float(np.interp(s, cum, path[:, 0])), float(np.interp(s, cum, path[:, 1]))


def stroke(E, pts, col, width):
    if pts is None or len(pts) < 2:
        return
    cv2.polylines(E, [np.round(pts * 16).astype(np.int32)], False, tuple(int(v) for v in col), int(width),
                  cv2.LINE_AA, shift=4)


def flicker(n, seed, length=9):
    """Deterministic neon buzz-on: a stuttering 0..1 sequence that ends fully on."""
    rng = np.random.default_rng(seed)
    seq = [0.25, 1.0, 0.0, 0.8, 0.15, 1.0, 0.5, 1.0, 0.9]
    seq = [min(1.0, max(0.0, v + rng.uniform(-0.12, 0.12))) for v in seq][:length]
    return seq[n] if 0 <= n < len(seq) else (1.0 if n >= len(seq) else 0.0)


def neon_light(mask, col, core_px):
    """Emissive light (RGB float) of a neon tube mask: white-hot core, coloured body, tight and wide glow."""
    col = np.float32(col)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * core_px + 1, 2 * core_px + 1))
    core = cv2.erode(mask, k)
    hot = col * 0.35 + 255 * 0.65
    light = mask[..., None] * col * 0.85 + core[..., None] * (hot - col * 0.85)
    g1 = cv2.GaussianBlur(mask, (0, 0), 7)[..., None] * col * 1.5
    h, w = mask.shape
    small = cv2.resize(mask, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    g2 = cv2.resize(cv2.GaussianBlur(small, (0, 0), 7), (w, h))[..., None] * col * 1.1
    g3 = cv2.resize(cv2.GaussianBlur(small, (0, 0), 22), (w, h))[..., None] * col * 0.9
    return light + g1 + g2 + g3


class Style:
    def __init__(self, clip, opts):
        self.clip = clip
        self.P = P = Plan(clip, opts)
        self.fr = c.Framing(clip, 1215)
        self.N = min(clip.n, len(clip.face))
        self.dur = self.N / clip.fps
        fb = np.array([self.fr.face_box(i) for i in range(0, self.N, 3)])
        self.head_top = float(np.median(fb[:, 1] - 0.95 * fb[:, 3]))  # top of the hair
        self.cam = fx.Cam(self.fr, self.camera_keys(), drift=3.0)
        self.signs = {}

    def camera_keys(self):
        B = self.P.beats
        keys = [{"t": 0.0, "zoom": 1.12}, {"t": 0.0, "zoom": 1.0, "ease": "smooth", "dur": 0.95}]
        for j, b in enumerate(B):
            kt = max(b["key_t"], 0.5 if j == 0 else 0)
            nx = B[j + 1]["key_t"] - 0.05 if j + 1 < len(B) else 99
            m = j % 4
            if j == len(B) - 1 and j > 0:
                m = 3
            if m == 0:
                keys += [{"t": kt, "zoom": 1.05, "ease": "spring", "freq": 1.5, "damp": 0.62}]
            elif m == 1:
                keys += [{"t": kt, "zoom": 1.08, "rot": 1.6, "ease": "spring", "freq": 1.4, "damp": 0.55},
                         {"t": min(kt + 0.7, nx), "rot": 0.0, "ease": "smooth", "dur": 0.8}]
            elif m == 2:
                keys += [{"t": kt, "zoom": 1.15, "ease": "spring", "freq": 2.4, "damp": 0.45},
                         {"t": min(kt + 0.4, nx), "zoom": 1.08, "ease": "smooth", "dur": 0.6}]
            else:
                keys += [{"t": kt, "zoom": 1.0, "rot": 0.0, "ease": "expo", "dur": 0.6}]
        return keys

    # ---- the neon sign on the wall ----
    def sign(self, k):
        if k in self.signs:
            return self.signs[k]
        b = self.P.beats[k]
        col = SIGN_COLS[k % len(SIGN_COLS)]
        lab = b["label"].upper()
        trk, wg = 0.08, 300
        size = min(230, c.fit_size(lab, "montserrat", 960, wg, trk))
        lines = [lab]
        ws = lab.split()
        if size < 150 and len(ws) > 1:
            cut = min(range(1, len(ws)), key=lambda q: abs(len(" ".join(ws[:q])) - len(" ".join(ws[q:]))))
            lines = [" ".join(ws[:cut]), " ".join(ws[cut:])]
            size = min(200, min(c.fit_size(ln, "montserrat", 960, wg, trk) for ln in lines))
        lh = size * 1.02
        base_last = self.head_top + 0.12 * size
        y_top = int(base_last - lh * (len(lines) - 1) - size - 200)
        y_top = max(-120, y_top)
        rh = int(base_last + size * 0.35 + 200) - y_top
        mask = np.zeros((rh, c.W), np.float32)
        broken = np.zeros((rh, c.W), np.float32)
        f = c.font("montserrat", int(size), wg)
        for li, ln in enumerate(lines):
            p, base = c.text_patch(ln, "montserrat", int(size), (255, 255, 255), wg, trk)
            x0 = int(round(c.W / 2 - p.shape[1] / 2))
            y0 = int(round(base_last - lh * (len(lines) - 1 - li) - base - y_top))
            al = p[..., 3] / 255
            hh, ww = al.shape
            ys, xs = max(0, y0), max(0, x0)
            ye, xe = min(rh, y0 + hh), min(c.W, x0 + ww)
            mask[ys:ye, xs:xe] = np.maximum(mask[ys:ye, xs:xe], al[ys - y0:ye - y0, xs - x0:xe - x0])
            if li == len(lines) - 1 and len(ln) > 2:  # one letter that buzzes like a tired tube
                q = max(1, len(ln) - 3) if ln[max(1, len(ln) - 3)] != " " else 1
                xq = sum(f.getlength(ch) for ch in ln[:q]) + trk * size * q
                lp, _ = c.text_patch(ln[q], "montserrat", int(size), (255, 255, 255), wg, 0.0)
                la = lp[..., 3] / 255
                lx = int(round(x0 + xq))
                lys, lxs = max(0, y0), max(0, lx)
                lye, lxe = min(rh, y0 + la.shape[0]), min(c.W, lx + la.shape[1])
                broken[lys:lye, lxs:lxe] = la[lys - y0:lye - y0, lxs - lx:lxe - lx]
        core = max(2, int(size * 0.02))
        light = neon_light(mask, col, core)
        bl = neon_light(broken * mask, col, core)
        self.signs[k] = (light, bl, y_top, col)
        return self.signs[k]

    def sign_level(self, k, i, t):
        """Brightness of sign k at frame i: buzz on at its key, stutter off when the next idea arrives."""
        B = self.P.beats
        fps = self.clip.fps
        on = B[k]["key_t"] - 0.04
        off = B[k + 1]["key_t"] - 0.04 if k + 1 < len(B) else 99
        if t < on or t >= off + 0.2:
            return 0.0, 1.0
        n = int(math.floor((t - on) * fps))
        lv = flicker(n, 31 + k)
        if t >= off:
            m = int(math.floor((t - off) * fps))
            lv *= [0.7, 0.0, 0.5, 0.0, 0.0][min(m, 4)]
        lv *= 0.95 + 0.05 * math.sin(i * 2.7 + k)
        bt = on + 0.55
        bseq = [1, 0.15, 1, 0.0, 0.6, 0.05, 1]
        bm = int(math.floor((t - bt) * fps))
        bro = bseq[bm] if 0 <= bm < len(bseq) else 1.0
        return lv, bro

    # ---- frame ----
    def frame(self, i, rgb, alpha):
        t = i / self.clip.fps
        fps = self.clip.fps
        B = self.P.beats
        img, a = self.cam.place(i, t, rgb, alpha)
        p = self.cam.at(t)
        fcx, fcy = self.fr.face_box(i)[:2]
        # tube power: races on with a flicker, surges on the payoff
        race = c.lerp(0.14, 1.0, c.ease_out_cubic(c.prog(t, 0.0, 0.52)))
        fl = [1.0, 0.3, 0.95, 0.12, 1.0, 0.5, 1.0]
        pw = fl[i] if i < len(fl) else 1.0
        last = B[-1]["key_t"] if len(B) > 1 else 99
        surge = 0.55 * math.exp(-max(t - last, 0) / 0.3) if t >= last else 0.0
        surge += 0.7 * math.exp(-t / 0.12)  # ignition burst
        # grade: wall falls into darkness, speaker stays lit (more so once the tube is on)
        lit = c.lerp(0.42, 0.95, min(1.0, race) * (0.4 + 0.6 * pw))
        a3 = cv2.GaussianBlur(a, (0, 0), 2)[..., None]
        x = img / 255
        lum = x @ np.float32([0.299, 0.587, 0.114])
        wall = (lum[..., None] + (x - lum[..., None]) * 0.35) * np.float32([0.26, 0.28, 0.37])
        him = lum[..., None] + (x - lum[..., None]) * 0.86
        him = np.clip((him - 0.5) * 1.12 + 0.5, 0, 1) * lit * np.float32([0.97, 0.98, 1.03])
        img = (him * a3 + wall * (1 - a3)) * 255
        # the sign(s) on the wall behind him (matte sandwich: he occludes it), moving with the camera
        z = p["zoom"]
        M = np.float32([[z, 0, fcx - z * fcx + p["dx"]], [0, z, fcy - z * fcy + p["dy"]]])
        S = None
        for k in range(len(B)):
            lv, bro = self.sign_level(k, i, t)
            if lv <= 0.01:
                continue
            light, bl, y_top, col = self.sign(k)
            Lk = light * lv - bl * (lv * (1 - bro))
            Mk = M.copy()
            Mk[:, 2] += M[:, 1] * y_top
            Wk = cv2.warpAffine(Lk, Mk, (c.W, c.H), flags=cv2.INTER_LINEAR, borderValue=0)
            S = Wk if S is None else S + Wk
        if S is not None:
            img += S * (1 - a3)
            # its glow wraps onto his hair and shoulders
            small = cv2.resize(S, (HW // 2, HH // 2), interpolation=cv2.INTER_AREA)
            wrap = cv2.resize(cv2.GaussianBlur(small, (0, 0), 6), (c.W, c.H))
            edge = a3 * (1 - cv2.GaussianBlur(a, (0, 0), 10)[..., None])
            img += wrap * edge * 2.2 + wrap * a3 * 0.12
        # the neon outline
        path = silhouette(a)
        if path is not None and len(path) > 10:
            top = int(np.argmin(path[:, 1]))
            A, Bp = path[:top + 1], path[top:][::-1]
            E = np.zeros((c.H, c.W, 3), np.uint8)
            cen = np.float32([fcx, fcy + 220])
            tube = PINK
            k2 = 1 if len(B) > 1 else 99
            t2 = B[k2]["key_t"] if k2 < len(B) else 99
            pulse = c.prog(t, t2 - 0.05, 0.5)
            if t >= t2 - 0.05 + 0.5:
                tube = CYAN
            # shockwave echo on the big word
            k3 = 2 if len(B) > 3 else None
            if k3 is not None:
                e = c.prog(t, B[k3]["key_t"] - 0.02, 0.6)
                if 0 < e < 1:
                    s = 1 + 0.32 * fx.expo(t, B[k3]["key_t"] - 0.02, 0.6)
                    cc = np.float32(SIGN_COLS[k3 % 4]) * (1 - e) ** 1.3
                    for half in (A, Bp):
                        stroke(E, cen + (half - cen) * s, cc, 7)
                    s2 = 1 + 0.16 * fx.expo(t, B[k3]["key_t"] + 0.06, 0.6)
                    for half in (A, Bp):
                        stroke(E, cen + (half - cen) * s2, cc * 0.7, 4)
            # double tube on the payoff
            if t >= last:
                d = fx.spring(t, last, 1.8, 0.6)
                for half in (A, Bp):
                    stroke(E, cen + (half - cen) * (1 + 0.045 * d), np.float32(PINK) * min(1, d * 1.5), 6)
            for half in (A, Bp):
                if 0 < pulse < 1:  # colour change travels up the tube as an energy pulse
                    stroke(E, sub(half, 0, min(race, pulse)), CYAN, 8)
                    stroke(E, sub(half, pulse, race), PINK, 8)
                    stroke(E, sub(half, max(0, pulse - 0.05), pulse), (255, 255, 255), 14)
                else:
                    stroke(E, sub(half, 0, race), tube, 8)
                stroke(E, sub(half, 0, race), (255, 240, 250), 3)
                if race < 0.995:  # comet heads while they race
                    hx, hy = tip(half, race)
                    cv2.circle(E, (int(hx), int(hy)), 9, (255, 255, 255), -1, cv2.LINE_AA)
            Ef = E.astype(np.float32)
            g1 = cv2.GaussianBlur(Ef, (0, 0), 6)
            small = cv2.resize(Ef, (HW, HH), interpolation=cv2.INTER_AREA)
            g2 = cv2.resize(cv2.GaussianBlur(small, (0, 0), 14), (c.W, c.H))
            light = Ef * 0.9 + g1 * 1.6 + g2 * 1.5
            img += light * pw * (1 + surge)
        img = c.bloom(img, 165, 22, 0.45 + surge * 0.6)
        img = c.vignette(img, 0.45)
        return c.grain(img, i, 3.0)
