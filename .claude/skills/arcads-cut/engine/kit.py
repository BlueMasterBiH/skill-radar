"""Edit kit shared by the styles: camera rig, captions, live crops, frames, arrows, icons, graphic shapes, looks.

Everything is a pure function of time and inputs, so frames are repeatable.
"""
import math
from functools import lru_cache

import cv2
import numpy as np

import core as c



# ---------- camera rig ----------

class Rig:
    """Keyframed virtual camera over a Framing. keys: list of dicts with t and any of
    zoom, dx, dy (canvas px), rot (deg), ease (seconds; 0 = hard cut on the key)."""

    def __init__(self, fr, keys, shakes=(), whips=()):
        self.fr = fr
        self.keys = sorted(keys, key=lambda k: k["t"])
        self.shakes = shakes  # (t0, dur, amp px)
        self.whips = whips  # (t, dir) : 3-frame horizontal smear around t
        self.rng = np.random.default_rng(17)
        self.noise = self.rng.normal(0, 1, (4000, 3))

    def at(self, t):
        cur = {"zoom": 1.0, "dx": 0.0, "dy": 0.0, "rot": 0.0}
        prev = dict(cur)
        for k in self.keys:
            if k["t"] > t:
                break
            prev = dict(cur)
            for name in ("zoom", "dx", "dy", "rot"):
                if name in k:
                    cur[name] = k[name]
            e = k.get("ease", 0)
            if e and t < k["t"] + e:
                x = c.ease_in_out_cubic((t - k["t"]) / e)
                cur = {n: c.lerp(prev[n], cur[n], x) for n in cur}
        sx = sy = 0.0
        for t0, dur, amp in self.shakes:
            if t0 <= t < t0 + dur:
                f = int(t * 50) % 4000
                fall = 1 - (t - t0) / dur
                sx += self.noise[f, 0] * amp * fall
                sy += self.noise[f, 1] * amp * fall
        cur["dx"] += sx
        cur["dy"] += sy
        return cur

    def place(self, i, t, rgb, alpha):
        p = self.at(t)
        fb = self.fr.clip.face[min(i, self.fr.clip.n - 1)]
        zc = (fb[0] - self.fr.x0, fb[1] - self.fr.y0)
        img, a = self.fr.place_t(rgb, alpha, p["zoom"], zc, p["dx"], p["dy"], p["rot"])
        for tw, d in self.whips:
            k = abs(t - tw)
            if k < 0.09:
                img = smear(img, int(90 * (1 - k / 0.09)) * d)
        return img, a


def smear(img, px):
    if abs(px) < 2:
        return img
    k = np.zeros((1, abs(px)), np.float32)
    k[:] = 1.0 / abs(px)
    return cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT)


# ---------- captions ----------

def phrase_groups(words, breaks=("stack", "put", "and", "in"), max_words=3):
    """Split words into short caption groups at phrase breaks; at most max_words per group."""
    out, cur = [], []
    for k, w in enumerate(words):
        key = norm(w["w"])
        if cur and ((key in breaks and k > 2) or len(cur) >= max_words):
            out.append(cur)
            cur = []
        cur.append(w)
    if cur:
        out.append(cur)
    return out


def norm(s):
    return "".join(ch for ch in s.lower() if ch.isalnum())


def current(groups, t):
    for k, g in enumerate(groups):
        end = groups[k + 1][0]["s"] if k + 1 < len(groups) else 1e9
        if g[0]["s"] - 0.04 <= t < end - 0.04:
            return g
    return None


def caption_line(dst, group, t, x, y, key="grotesk", size=80, wght=800, color=(255, 255, 255), hot=(),
                 hot_color=(255, 214, 0), upper=False, stroke=0, stroke_color=(0, 0, 0), mode="pop", gap=0.26,
                 align="m", shadow=0.35, box=(124, 242, 154), box_text=(10, 10, 10), dim=1.0, tracking=0.0):
    """Draw one caption group laid out on a line. Modes:
    pop - each word scales in on its start; karaoke - all words shown, the spoken one boxed;
    highlighter - keywords get a marker swipe; rise - words rise and fade in."""
    words = [(w["w"].upper() if upper else w["w"]) for w in group]
    ws = [c.text_w(s, key, size, wght, tracking) for s in words]
    total = sum(ws) + gap * size * (len(words) - 1)
    x0 = {"m": x - total / 2, "l": x, "r": x - total}[align]
    xs = []
    for ww in ws:
        xs.append(x0)
        x0 += ww + gap * size
    for w, s, ww, xx in zip(group, words, ws, xs):
        is_hot = norm(w["w"]) in hot
        col = hot_color if is_hot else color
        if mode == "karaoke":
            on = w["s"] - 0.03 <= t < w["e"] + 0.05
            if on:
                bw, bh = int(ww + size * 0.36), int(size * 1.14)
                c.over(dst, c.fill(c.rrect(bw, bh, int(size * 0.22)), box), xx - size * 0.18, y - size * 0.9)
                col = box_text
            elif t < w["s"]:
                col = tuple(int(v * 0.55) for v in color)
            c.put_text(dst, s, key, size, xx, y, col, wght, tracking=tracking, stroke=stroke,
                       stroke_color=stroke_color)
            continue
        p = c.prog(t, w["s"] - 0.05, 0.14)
        if p <= 0:
            continue
        sc, dy, op = 1.0, 0.0, 1.0
        if mode == "pop":
            sc = c.ease_out_back(p, 2.2) * 0.35 + 0.65
            op = min(1, p * 3)
        elif mode == "rise":
            dy = (1 - c.ease_out_cubic(p)) * 18
            op = p
        if mode == "highlighter" and is_hot:
            sw = c.ease_out_cubic(c.prog(t, w["s"] - 0.09, 0.08))
            marker(dst, xx - size * 0.12, y - size * 0.62, (ww + size * 0.24) * sw, size * 0.74, hot_color, seed=len(s))
            col = box_text
        if shadow and not stroke and not (mode == "highlighter" and is_hot):
            c.put_text(dst, s, key, size, xx + 3, y + 6 + dy, (0, 0, 0), wght, tracking=tracking,
                       opacity=shadow * op, blur=6)
        if sc != 1.0:
            cx = xx + ww / 2
            c.put_text(dst, s, key, size * sc, cx, y + dy - size * 0.35 * (1 - sc), col, wght, tracking=tracking,
                       anchor="ms", opacity=op * dim, stroke=int(stroke * sc), stroke_color=stroke_color)
        else:
            c.put_text(dst, s, key, size, xx, y + dy, col, wght, tracking=tracking, opacity=op * dim,
                       stroke=stroke, stroke_color=stroke_color)


def marker(dst, x, y, w, h, color, seed=0, a=1.0):
    """A rough highlighter swipe: slightly uneven ends and edges."""
    if w < 4:
        return
    w, h = int(w), int(h)
    rng = np.random.default_rng(seed)
    m = np.zeros((h + 8, w + 8), np.float32)
    top = 4 + rng.normal(0, 1.5, 6).cumsum() * 0.4
    pts = [(4, 6), *[(4 + w * k / 5, top[k]) for k in range(6)], (w + 4, h), *[(4 + w * (5 - k) / 5, h + 2 - abs(top[k]) * 0.5) for k in range(6)]]
    cv2.fillPoly(m, [np.array(pts, np.int32)], 1.0, cv2.LINE_AA)
    m = cv2.GaussianBlur(m, (0, 0), 0.8)
    c.over(dst, c.fill(m, color, a), x - 4, y - 4)


# ---------- footage ----------



def face_crop(clip, rgb, i, w, h, span=2.6, face_at=0.42, alpha=None):
    """A live crop of the clip around the face, w x h (span = crop width in face widths). The second camera."""
    fx, fy, fw, fh = clip.face[min(i, clip.n - 1)]
    cw = min(clip.sw, fw * span)
    ch = cw * h / w
    if ch > clip.sh:
        ch = clip.sh
        cw = ch * w / h
    x0 = int(c.clamp(fx - cw / 2, 0, clip.sw - cw))
    y0 = int(c.clamp(fy - ch * face_at, 0, clip.sh - ch))
    img = cv2.resize(rgb[y0:y0 + int(ch), x0:x0 + int(cw)], (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    if alpha is None:
        return img
    a = cv2.resize(alpha[y0:y0 + int(ch), x0:x0 + int(cw)], (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)
    return img, a / 255


def framed(img, radius=0, border=0, border_color=(255, 255, 255), pad_bottom=0):
    """RGB float -> premultiplied RGBA patch with optional rounded corners and a border (polaroid via pad_bottom)."""
    h, w = img.shape[:2]
    if border or pad_bottom:
        W2, H2 = w + 2 * border, h + 2 * border + pad_bottom
        base = np.empty((H2, W2, 3), np.float32)
        base[:] = border_color
        base[border:border + h, border:border + w] = img
        img, (h, w) = base, (H2, W2)
    m = c.rrect(w, h, radius) if radius else np.ones((h, w), np.float32)
    return np.concatenate([img * m[..., None], m[..., None] * 255], -1)


def rotated(patch, deg, scale=1.0):
    h, w = patch.shape[:2]
    pad = int(math.hypot(w, h) * scale / 2 - min(w, h) * scale / 2) + 4
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, scale)
    M[:, 2] += pad
    return cv2.warpAffine(patch, M, (w + 2 * pad, h + 2 * pad), flags=cv2.INTER_LINEAR)


def place(dst, patch, cx, cy, deg=0.0, scale=1.0, opacity=1.0, shadow_a=0.0, shadow_blur=18, shadow_dy=12):
    """Place a patch centred at (cx, cy), rotated and scaled, with an optional soft shadow."""
    if scale <= 0.01 or opacity <= 0:
        return
    p = rotated(patch, deg, scale) if (deg or scale != 1.0) else patch
    if shadow_a:
        sh, (ox, oy) = c.shadow(p, shadow_blur, shadow_a, shadow_dy)
        c.over(dst, sh, cx - p.shape[1] / 2 + ox, cy - p.shape[0] / 2 + oy, opacity)
    c.over(dst, p, cx - p.shape[1] / 2, cy - p.shape[0] / 2, opacity)


def check_badge(size, color=(52, 199, 89)):
    p = c.fill(c.rrect(size, size, size // 2), color)
    c.polyline(p, [(size * 0.28, size * 0.52), (size * 0.44, size * 0.68), (size * 0.74, size * 0.34)],
               (255, 255, 255), max(3, size // 11))
    return p


def burst_word(text, size, fill=(255, 214, 0), ink=(12, 12, 12), seed=0):
    """A comic starburst with a word in it, as a patch."""
    tw = c.text_w(text, "archivo", size)
    r1 = max(tw * 0.75, size * 1.6)
    d = int(r1 * 2.3)
    p = np.zeros((d, d, 4), np.float32)
    poly_fill(p, burst_pts(d / 2, d / 2, r1 * 0.72, r1, 12, seed), fill, 1, ink, max(4, size // 10))
    c.put_text(p, text, "archivo", size, d / 2, d / 2 + size * 0.36, ink, anchor="ms")
    return p


def note(text, w=300, h=260, color=(255, 236, 140), ink=(30, 28, 25), size=44):
    """A sticky note with marker handwriting (wraps to fit)."""
    p = c.fill(np.ones((h, w), np.float32), color)
    p[:22, :, :3] *= 0.93
    words, lines, cur = text.split(), [], ""
    for wd in words:
        if c.text_w((cur + " " + wd).strip(), "marker", size) > w - 40:
            lines.append(cur)
            cur = wd
        else:
            cur = (cur + " " + wd).strip()
    lines.append(cur)
    y = h / 2 - (len(lines) - 1) * size * 0.6 + size * 0.35
    for ln in lines:
        c.put_text(p, ln, "marker", size, w / 2, y, ink, anchor="ms")
        y += size * 1.2
    return p


def pop_lines(dst, cx, cy, side, t, t0, color=(255, 255, 255), length=70, width=8):
    """Three short emphasis strokes radiating beside a point, growing then fading."""
    p = c.prog(t, t0, 0.35)
    if p <= 0 or p >= 1:
        return
    grow = c.ease_out_cubic(min(1, p * 2.2))
    fade = 1 - max(0, (p - 0.5) * 2)
    for ang in (-35, 0, 35):
        a = math.radians(ang + (0 if side > 0 else 180))
        r0 = 20
        p0 = (cx + r0 * math.cos(a), cy + r0 * math.sin(a))
        p1 = (cx + (r0 + length * grow) * math.cos(a), cy + (r0 + length * grow) * math.sin(a))
        c.polyline(dst, [p0, p1], color, width, 1.0, fade)


# ---------- shapes ----------

def arrow(dst, p0, p1, color, width=10, progress=1.0, bend=0.25, head=34, opacity=1.0, seed=0):
    """Curved hand-drawn-ish arrow from p0 to p1, drawn along its length by progress."""
    p0, p1 = np.array(p0, np.float32), np.array(p1, np.float32)
    mid = (p0 + p1) / 2
    n = np.array([-(p1 - p0)[1], (p1 - p0)[0]], np.float32)
    ctrl = mid + n * bend
    ts = np.linspace(0, 1, 40)[:, None]
    pts = (1 - ts) ** 2 * p0 + 2 * (1 - ts) * ts * ctrl + ts ** 2 * p1
    c.polyline(dst, pts, color, width, progress, opacity)
    if progress >= 0.98:
        d = pts[-1] - pts[-4]
        d = d / (np.linalg.norm(d) + 1e-6)
        left = pts[-1] - head * (d * math.cos(0.5) - np.array([-d[1], d[0]]) * math.sin(0.5))
        right = pts[-1] - head * (d * math.cos(0.5) + np.array([-d[1], d[0]]) * math.sin(0.5))
        c.polyline(dst, [left, pts[-1], right], color, width, 1.0, opacity)


def ellipse_pts(cx, cy, rx, ry, a0=-90, sweep=380, n=64, grow=0.06):
    a = np.radians(np.linspace(a0, a0 + sweep, n))
    g = np.linspace(1.0, 1.0 + grow, n)
    return np.stack([cx + rx * g * np.cos(a), cy + ry * g * np.sin(a)], 1)


def icon(dst, name, cx, cy, s, color, width=4, progress=1.0, opacity=1.0):
    """Line icons on an s x s box centred at (cx, cy): camera, bolt, stack, check, play, mic, spark."""
    u = s / 100
    P = lambda pts: [(cx + x * u, cy + y * u) for x, y in pts]
    if name == "camera":
        c.polyline(dst, P([(-42, -22), (42, -22), (42, 32), (-42, 32), (-42, -22)]), color, width, progress, opacity)
        c.polyline(dst, ellipse_pts(cx, cy + 5 * u, 17 * u, 17 * u, sweep=360, grow=0), color, width, progress, opacity)
        c.polyline(dst, P([(-18, -22), (-10, -34), (10, -34), (18, -22)]), color, width, progress, opacity)
    elif name == "bolt":
        c.polyline(dst, P([(8, -44), (-22, 6), (2, 6), (-8, 44), (24, -8), (0, -8), (8, -44)]), color, width, progress, opacity)
    elif name == "stack":
        for k in range(3):
            y = -26 + k * 22
            c.polyline(dst, P([(-36, y), (36, y), (36, y + 14), (-36, y + 14), (-36, y)]), color, width, progress, opacity)
    elif name == "check":
        c.polyline(dst, P([(-34, 2), (-10, 26), (36, -26)]), color, width, progress, opacity)
    elif name == "play":
        c.polyline(dst, P([(-22, -30), (30, 0), (-22, 30), (-22, -30)]), color, width, progress, opacity)
    elif name == "mic":
        c.polyline(dst, P([(-12, -40), (12, -40), (12, 4), (-12, 4), (-12, -40)]), color, width, progress, opacity)
        c.polyline(dst, P([(-26, -8), (-26, 10), (0, 24), (26, 10), (26, -8)]), color, width, progress, opacity)
        c.polyline(dst, P([(0, 24), (0, 40)]), color, width, progress, opacity)
    elif name == "spark":
        c.polyline(dst, P([(0, -40), (8, -8), (40, 0), (8, 8), (0, 40), (-8, 8), (-40, 0), (-8, -8), (0, -40)]), color, width, progress, opacity)


def burst_pts(cx, cy, r0, r1, n=14, seed=0):
    """Comic starburst polygon."""
    rng = np.random.default_rng(seed)
    pts = []
    for k in range(2 * n):
        a = math.pi * k / n
        r = r1 * (0.9 + 0.2 * rng.random()) if k % 2 == 0 else r0
        pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return np.array(pts, np.float32)


def poly_fill(dst, pts, color, a=1.0, outline=None, ow=6):
    pts = np.asarray(pts, np.float32)
    x0, y0 = np.floor(pts.min(0) - ow * 2).astype(int)
    x1, y1 = np.ceil(pts.max(0) + ow * 2).astype(int)
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    cv2.fillPoly(m, [((pts - [x0, y0]) * 16).astype(np.int32)], 255, cv2.LINE_AA, shift=4)
    c.over(dst, c.fill(m.astype(np.float32) / 255, color, a), x0, y0)
    if outline is not None:
        c.polyline(dst, np.vstack([pts, pts[:1]]), outline, ow, 1.0, a)


def speed_lines(dst, cx, cy, t, n=90, r0=380, color=(0, 0, 0), a=0.9, seed=3):
    """Radial manga speed lines, re-seeded every 2 frames."""
    rng = np.random.default_rng(seed + int(t * 12.5))
    for _ in range(n):
        ang = rng.uniform(0, 2 * math.pi)
        r = r0 * rng.uniform(0.85, 1.25)
        L = rng.uniform(300, 1300)
        w = rng.uniform(2, 9)
        p0 = (cx + r * math.cos(ang), cy + r * math.sin(ang))
        p1 = (cx + (r + L) * math.cos(ang), cy + (r + L) * math.sin(ang))
        c.polyline(dst, [p0, p1], color, w, 1.0, a)


def confetti(dst, t, t0, n=70, origin=(540, 700), seed=8, colors=((255, 214, 0), (255, 90, 90), (90, 200, 255), (124, 242, 154))):
    if t < t0:
        return
    rng = np.random.default_rng(seed)
    dt = t - t0
    for k in range(n):
        vx, vy = rng.normal(0, 520), rng.uniform(-1500, -700)
        x = origin[0] + vx * dt
        y = origin[1] + vy * dt + 2600 * dt * dt / 2
        if not (-40 < x < c.W + 40 and -40 < y < c.H + 40):
            continue
        w, h = int(rng.uniform(12, 22)), int(rng.uniform(22, 36))
        ang = rng.uniform(0, 360) + dt * rng.uniform(-600, 600)
        place(dst, c.fill(np.ones((h, w), np.float32), colors[k % len(colors)]), x, y, ang)


# ---------- looks ----------

def halftone(img, cell=8, ink=(20, 20, 20), paper=(245, 240, 228), angle=15):
    g = (img @ np.array([0.299, 0.587, 0.114], np.float32)) / 255
    h, w = g.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    Mi = cv2.invertAffineTransform(M)
    gr = cv2.warpAffine(g, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    small = cv2.resize(gr, (max(1, w // cell), max(1, h // cell)), interpolation=cv2.INTER_AREA)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    fx, fy = (xs % cell) - cell / 2 + 0.5, (ys % cell) - cell / 2 + 0.5
    d = np.sqrt(fx ** 2 + fy ** 2)
    dark = 1 - cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
    r = np.sqrt(np.clip(dark, 0, 1)) * cell * 0.62
    dots = np.clip(r - d + 0.5, 0, 1)
    dots = cv2.warpAffine(dots, Mi, (w, h), borderMode=cv2.BORDER_REFLECT)
    ink, paper = np.array(ink, np.float32), np.array(paper, np.float32)
    return paper * (1 - dots[..., None]) + ink * dots[..., None]


def ink_look(img, levels=4, edge=0.6):
    """Manga ink: posterised greys + bold edges + screentone in the mid tones."""
    g = cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    g = cv2.bilateralFilter(g, 7, 40, 7)
    q = np.floor(g / 256 * levels) / (levels - 1)
    edges = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 5).astype(np.float32) / 255
    out = q * 255
    mid = (q > 0.2) & (q < 0.8)
    h, w = g.shape
    ys, xs = np.mgrid[0:h, 0:w]
    tone = (((xs + ys) % 7) < 2) | (((xs - ys) % 7) < 2)
    out = np.where(mid & tone, out * 0.55, out)
    out = out * (1 - edge * (1 - edges))
    return np.repeat(out[..., None], 3, -1).astype(np.float32)


def duotone(img, lo, hi, contrast=1.2):
    g = (img @ np.array([0.299, 0.587, 0.114], np.float32)) / 255
    x = np.clip((g - 0.5) * contrast + 0.5, 0, 1)[..., None]
    return np.array(lo, np.float32) * (1 - x) + np.array(hi, np.float32) * x


def bw(img, contrast=1.15):
    g = img @ np.array([0.299, 0.587, 0.114], np.float32)
    g = np.clip((g / 255 - 0.5) * contrast + 0.5, 0, 1) * 255
    return np.repeat(g[..., None], 3, -1)


@lru_cache(maxsize=4)
def paper(h, w, color, fiber=0.0, seed=4):
    rng = np.random.default_rng(seed)
    base = np.array(color, np.float32) + rng.normal(0, 3.5, (h, w, 1)).astype(np.float32)
    if fiber:
        n = cv2.GaussianBlur(rng.normal(0, 1, (h // 4, w // 4)).astype(np.float32), (0, 0), 2)
        base += cv2.resize(n, (w, h))[..., None] * fiber
    return np.clip(base, 0, 255)


def torn_mask(w, h, seed=0, amp=7, edges="tb"):
    rng = np.random.default_rng(seed)
    m = np.ones((h, w), np.float32)
    for e in edges:
        prof = np.cumsum(rng.normal(0, 1.6, w if e in "tb" else h))
        prof = (prof - prof.min()) / (np.ptp(prof) + 1e-6) * amp + rng.integers(1, 3)
        prof = prof.astype(int)
        for k, d in enumerate(prof):
            if e == "t":
                m[:d, k] = 0
            elif e == "b":
                m[h - d:, k] = 0
            elif e == "l":
                m[k, :d] = 0
            else:
                m[k, w - d:] = 0
    return cv2.GaussianBlur(m, (0, 0), 0.7)


def tape(w=150, h=46, color=(240, 230, 200), a=0.82):
    m = torn_mask(w, h, seed=w, amp=5, edges="lr")
    p = c.fill(m, color, a)
    return p


def cutout(img, a, stroke=10, color=(255, 255, 255)):
    """Sticker cutout of the person: RGB + alpha -> premultiplied RGBA with a die-cut outline."""
    hard = (a > 0.5).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * stroke + 1, 2 * stroke + 1))
    ring = cv2.GaussianBlur(cv2.dilate(hard, k).astype(np.float32), (0, 0), 1.2)
    out = np.concatenate([np.array(color, np.float32) * ring[..., None], ring[..., None] * 255], -1)
    person = np.concatenate([img * a[..., None], a[..., None] * 255], -1)
    c.over(out, person)
    return out


def chroma(img, px=4):
    out = img.copy()
    out[..., 0] = np.roll(img[..., 0], px, 1)
    out[..., 2] = np.roll(img[..., 2], -px, 1)
    return out


def flash(img, amount):
    return img * (1 - amount) + 255 * amount


# ---------- timing helpers ----------

def pop(t, t0, dur=0.28, k=1.8):
    """Scale for a pop-in that starts at t0 (0 before)."""
    p = c.prog(t, t0, dur)
    return c.ease_out_back(p, k) if p > 0 else 0.0


def typed(text, t, t0, cps=28):
    """Text revealed character by character from t0."""
    n = int(max(0.0, t - t0) * cps)
    return text[:n]
