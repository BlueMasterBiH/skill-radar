"""Shared engine for the arcads-cut styles: framing, matte compositing, type, shapes, grade, output.

Every frame depends only on its index and the cached analysis (prep.py), so renders are repeatable.
Layers are premultiplied float32 RGBA arrays at canvas size (1080x1920).
"""
import json
import math
import subprocess
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1080, 1920
ENGINE = Path(__file__).resolve().parent
FONT_DIR = ENGINE.parent / "fonts"

FONTS = {
    "grotesk": ("InterTight-Variable.ttf", True),
    "mono": ("GeistMono-Variable.ttf", True),
    "serif": ("InstrumentSerif-Regular.ttf", False),
    "serif-i": ("InstrumentSerif-Italic.ttf", False),
    "display": ("PlayfairDisplay.ttf", True),
    "display-i": ("PlayfairDisplay-Italic.ttf", True),
    "marker": ("PermanentMarker-Regular.ttf", False),
    "anton": ("Anton-Regular.ttf", False),
    "archivo": ("ArchivoBlack-Regular.ttf", False),
    "montserrat": ("Montserrat.ttf", True),
    "spacemono": ("SpaceMono-Bold.ttf", False),
}


# ---------- timing ----------

def clamp(x, a=0.0, b=1.0):
    return a if x < a else b if x > b else x


def prog(t, start, dur):
    return clamp((t - start) / dur) if dur > 0 else float(t >= start)


def ease_out_expo(x):
    return 1.0 if x >= 1 else 1 - 2 ** (-10 * x)


def ease_in_expo(x):
    return 0.0 if x <= 0 else 2 ** (10 * x - 10)


def ease_out_cubic(x):
    return 1 - (1 - x) ** 3


def ease_in_out_cubic(x):
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ease_out_back(x, k=1.4):
    return 1 + (k + 1) * (x - 1) ** 3 + k * (x - 1) ** 2


def lerp(a, b, x):
    return a + (b - a) * x


# ---------- clip + analysis ----------

class Clip:
    def __init__(self, src, cache):
        self.src = str(src)
        self.cache = Path(cache)
        self.meta = json.load(open(self.cache / "meta.json"))
        self.fps = self.meta["fps"]
        self.n = self.meta["n"]
        self.sw, self.sh = self.meta["w"], self.meta["h"]
        self.words = json.load(open(self.cache / "words.json"))
        self.face = np.array(json.load(open(self.cache / "face.json")), np.float32)
        p = cv2.cvtColor(cv2.imread(str(self.cache / "plate.png")), cv2.COLOR_BGR2RGB)
        self.plate = cv2.resize(p, (self.sw, self.sh), interpolation=cv2.INTER_LINEAR)
        self.env = self._envelope()

    def _envelope(self):
        """Per-frame loudness 0..1 (for waveforms and audio-reactive graphics, even in silent renders)."""
        b = subprocess.run(["ffmpeg", "-v", "error", "-i", self.src, "-ac", "1", "-ar", "16000", "-f", "s16le", "-"],
                           capture_output=True).stdout
        x = np.frombuffer(b, np.int16).astype(np.float32) / 32768
        if len(x) == 0:
            return np.zeros(self.n, np.float32)
        hop = 16000 / self.fps
        rms = np.array([np.sqrt(np.mean(x[int(k * hop):int((k + 1) * hop)] ** 2) + 1e-9) for k in range(self.n)])
        return np.clip(rms / (np.percentile(rms, 95) + 1e-6), 0, 1).astype(np.float32)

    def word(self, text, nth=0):
        """Timing dict of the nth word whose letters start with `text` (case/punctuation-insensitive)."""
        key = "".join(c for c in text.lower() if c.isalnum())
        hits = [w for w in self.words if "".join(c for c in w["w"].lower() if c.isalnum()).startswith(key)]
        return hits[nth]

    def frames(self, start=0):
        p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", self.src, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             stdout=subprocess.PIPE, bufsize=self.sw * self.sh * 3)
        size = self.sw * self.sh * 3
        i = 0
        while i < self.n:
            buf = p.stdout.read(size)
            if len(buf) < size:
                break
            if i >= start:
                yield i, np.frombuffer(buf, np.uint8).reshape(self.sh, self.sw, 3)
            i += 1
        p.stdout.close()
        p.wait()

    def frame_at(self, i):
        out = subprocess.run(["ffmpeg", "-v", "error", "-i", self.src, "-vf", f"select=eq(n\\,{i})", "-vsync", "0",
                              "-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             capture_output=True, check=True).stdout
        return np.frombuffer(out, np.uint8).reshape(self.sh, self.sw, 3)

    def matte(self, i):
        return cv2.imread(str(self.cache / "matte" / f"{min(i, self.n - 1):04d}.png"), cv2.IMREAD_GRAYSCALE)


# ---------- framing ----------

class Framing:
    """Places a crop of the source into a canvas region (default: the whole 9:16 canvas).

    crop_w: width of the source crop, in pixels of a 2160-px-tall source (scaled to the real clip; smaller = tighter).
    The crop keeps full source height when it fits; otherwise it is cut around the face.
    When the scaled source is shorter than the region, the frame sits on the bottom and the wall is extended upward
    from the clean plate. When the clip is narrower than the requested crop (vertical phone footage asked for a
    wide shot), the frame is scaled down and the wall is extended on the sides too.
    region: (x, y, w, h) canvas rectangle to fill instead of the full canvas.
    """

    def __init__(self, clip, crop_w, region=(0, 0, W, H), cx=None, head_at=0.36, ext_dark=0.5):
        self.clip = clip
        rx, ry, rw, rh = region
        med_x = float(np.median(clip.face[:, 0]))
        k = clip.sh / 2160.0
        req = crop_w * k
        if cx is not None:
            cx = med_x + (cx - med_x) * k
        self.region = region
        self.cw = int(min(req, clip.sw))
        self.s = rw / req
        self.pw = rw if req <= clip.sw else int(round(self.cw * self.s))  # placed width
        self.mx = (rw - self.pw) // 2  # side margins filled with extended wall
        if cx is not None and self.pw < rw:  # narrow clip: slide the frame the way the style asked for, within the region
            self.mx = int(clamp(self.mx - (cx - med_x) * self.s, 0, rw - self.pw))
        self.cx = med_x if cx is None else cx
        self.x0 = int(round(clamp(self.cx - self.cw / 2, 0, clip.sw - self.cw)))
        full_h = clip.sh * self.s
        if full_h >= rh:  # crop vertically, keep the face at head_at of the region height
            ch = int(round(rh / self.s))
            fy = float(np.median(clip.face[:, 1]))
            self.y0 = int(round(clamp(fy - head_at * ch, 0, clip.sh - ch)))
            self.ch, self.ext = ch, 0
        else:
            self.y0, self.ch = 0, clip.sh
            self.ext = int(round(rh - full_h))
        self.ph = rh - self.ext  # placed height inside the region
        self.bg = self._background(ext_dark) if (self.ext or self.mx) else None

    def _background(self, dark):
        """Region-sized wall: the clean plate where the frame sits, stretched and darkened into the margins."""
        rx, ry, rw, rh = self.region
        plate = self.clip.plate[self.y0:self.y0 + self.ch, self.x0:self.x0 + self.cw]
        plate = cv2.resize(plate, (self.pw, self.ph), interpolation=cv2.INTER_AREA).astype(np.float32)
        row = np.empty((self.ph, rw, 3), np.float32)
        row[:, self.mx:self.mx + self.pw] = plate
        if self.pw < rw:
            side_dark = 0.5 + 0.5 * dark
            for side, width in (("l", self.mx), ("r", rw - self.mx - self.pw)):
                if width <= 0:
                    continue
                reps = int(np.ceil(width / self.pw))
                src = np.concatenate([plate[:, ::-1]] * reps, 1)  # mirror of the frame's edge
                strip = src[:, :width] if side == "r" else src[:, -width:]
                strip = cv2.GaussianBlur(strip, (0, 0), 24)
                xs = np.linspace(0, 1, width, dtype=np.float32)[None, :, None]  # 0 outer edge, 1 at the seam
                fall = side_dark + (1 - side_dark) * xs ** 0.7
                if side == "l":
                    row[:, :width] = strip * fall
                else:
                    row[:, rw - width:] = strip * fall[:, ::-1]
        if self.ext:
            band = cv2.GaussianBlur(row[:24].mean(0, keepdims=True), (0, 0), sigmaX=30)
            ys = np.linspace(0, 1, self.ext, dtype=np.float32)[:, None, None]  # 0 at top, 1 at the seam
            top = np.repeat(band, self.ext, 0) * (dark + (1 - dark) * ys ** 0.8)
            bg = np.concatenate([top, row], 0)
        else:
            bg = row
        bg = cv2.GaussianBlur(bg, (0, 0), 6)  # softens the seams between plate, sides and top
        rng = np.random.default_rng(7)
        return bg + rng.normal(0, 1.2, bg.shape).astype(np.float32)  # breaks banding

    def face_box(self, i):
        """Face [cx, cy, w, h] in canvas pixels."""
        fx, fy, fw, fh = self.clip.face[min(i, self.clip.n - 1)]
        rx, ry, _, _ = self.region
        return np.array([rx + self.mx + (fx - self.x0) * self.s, ry + self.ext + (fy - self.y0) * self.s,
                         fw * self.s, fh * self.s], np.float32)

    def place_t(self, rgb, alpha, zoom=1.0, zc=None, dx=0.0, dy=0.0, rot=0.0):
        """Like place(), plus a canvas-pixel offset and a rotation (degrees) about zc."""
        if not (dx or dy or rot):
            return self.place(rgb, alpha, zoom, zc)
        crop = rgb[self.y0:self.y0 + self.ch, self.x0:self.x0 + self.cw]
        acrop = alpha[self.y0:self.y0 + self.ch, self.x0:self.x0 + self.cw]
        zx, zy = zc if zc is not None else (self.cw / 2, self.ch / 2)
        M = cv2.getRotationMatrix2D((float(zx), float(zy)), rot, zoom)
        M[0, 2] += dx / self.s
        M[1, 2] += dy / self.s
        crop = cv2.warpAffine(crop, M, (self.cw, self.ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        acrop = cv2.warpAffine(acrop, M, (self.cw, self.ch), flags=cv2.INTER_LINEAR)
        return self.place(crop, acrop, 1.0, None, _cropped=True)

    def place(self, rgb, alpha, zoom=1.0, zc=None, _cropped=False):
        """Return (rgb float32 [0..255], alpha float32 [0..1]) for the region, optionally zoomed about zc."""
        rx, ry, rw, rh = self.region
        if _cropped:
            crop, acrop = rgb, alpha
        else:
            crop = rgb[self.y0:self.y0 + self.ch, self.x0:self.x0 + self.cw]
            acrop = alpha[self.y0:self.y0 + self.ch, self.x0:self.x0 + self.cw]
        if zoom != 1.0:
            zx, zy = zc if zc is not None else (self.cw / 2, self.ch / 2)
            M = np.float32([[zoom, 0, zx - zoom * zx], [0, zoom, zy - zoom * zy]])
            crop = cv2.warpAffine(crop, M, (self.cw, self.ch), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            acrop = cv2.warpAffine(acrop, M, (self.cw, self.ch), flags=cv2.INTER_LINEAR)
        img = cv2.resize(crop, (self.pw, self.ph), interpolation=cv2.INTER_AREA).astype(np.float32)
        a = cv2.resize(acrop, (self.pw, self.ph), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
        if self.bg is None:
            return img, a
        out = self.bg.copy()
        e, m, pw = self.ext, self.mx, self.pw
        r = rw - m - pw
        if m or r:  # narrow clip: the margins are a blurred mirror of the live frame (blur fill)
            small = cv2.resize(img, (max(1, pw // 4), max(1, self.ph // 4)), interpolation=cv2.INTER_AREA)
            for width, left in ((m, True), (r, False)):
                if width <= 0:
                    continue
                reps = int(np.ceil(width / pw)) + 1
                src = np.concatenate([small[:, ::-1]] * reps, 1)
                w4 = max(1, int(np.ceil(width / 4)))
                strip = src[:, -w4:] if left else src[:, :w4]
                strip = cv2.resize(cv2.GaussianBlur(strip, (0, 0), 6), (width, self.ph), interpolation=cv2.INTER_LINEAR)
                xs = np.linspace(0.78, 1.0, width, dtype=np.float32)[None, :, None]
                if left:
                    out[e:, :width] = strip * xs
                else:
                    out[e:, rw - width:] = strip * xs[:, ::-1]
        win = out[e:, m:m + pw]
        base = win.copy()
        win[:] = img
        if e:  # feather the top seam into the extended wall
            f = 40
            wgt = np.linspace(0, 1, f, dtype=np.float32)[:, None, None]
            win[:f] = base[:f] * (1 - wgt) + img[:f] * wgt
        if m or r:  # and the side seams into the blur fill
            f = 36
            wgt = np.linspace(0, 1, f, dtype=np.float32)[None, :, None]
            if m:
                win[:, :f] = out[e:, m - 1:m] * (1 - wgt) + img[:, :f] * wgt
            if r:
                win[:, -f:] = out[e:, m + pw:m + pw + 1] * wgt[:, ::-1] + img[:, -f:] * (1 - wgt[:, ::-1])
        aa = np.zeros((rh, rw), np.float32)
        aa[e:, m:m + pw] = a
        return out, aa


# ---------- layers ----------

def layer(h=H, w=W):
    return np.zeros((h, w, 4), np.float32)


def rgba(color, a=1.0):
    c = np.array(color[:3], np.float32)
    return np.array([*(c * a), 255 * a], np.float32)


def over(dst, src, x=0, y=0, opacity=1.0):
    """Composite premultiplied patch src (h,w,4) over dst at (x,y) with clipping. dst may be RGB or RGBA."""
    h, w = src.shape[:2]
    x, y = int(round(x)), int(round(y))
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, dst.shape[1]), min(y + h, dst.shape[0])
    if x0 >= x1 or y0 >= y1 or opacity <= 0:
        return dst
    s = src[y0 - y:y1 - y, x0 - x:x1 - x] * opacity
    d = dst[y0:y1, x0:x1]
    inv = 1 - s[..., 3:4] / 255
    if dst.shape[2] == 4:
        d[:] = s + d * inv
    else:
        d[:] = s[..., :3] + d * inv
    return dst


def composite(plate, alpha, back=None, front=None):
    """Person-aware composite: back layer goes behind the person, front layer over everything."""
    out = plate.copy()
    if back is not None:
        bg = plate * (1 - back[..., 3:4] / 255) + back[..., :3]
        out = plate * alpha[..., None] + bg * (1 - alpha[..., None])
    if front is not None:
        out = out * (1 - front[..., 3:4] / 255) + front[..., :3]
    return out


@lru_cache(maxsize=64)
def rrect(w, h, r, feather=1.0):
    """Anti-aliased rounded-rectangle alpha mask [0..1], shape (h, w)."""
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    qx = np.abs(xs + 0.5 - w / 2) - (w / 2 - r)
    qy = np.abs(ys + 0.5 - h / 2) - (h / 2 - r)
    d = np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r
    return np.clip(0.5 - d / feather, 0, 1)


def fill(mask, color, a=1.0):
    """Premultiplied RGBA patch of a mask filled with a colour."""
    m = (mask * a)[..., None]
    c = np.array(color[:3], np.float32)
    return np.concatenate([m * c, m * 255], -1).astype(np.float32)


def shadow(patch, blur=24, a=0.45, dy=10, pad=None):
    """A soft drop shadow for a patch, returned as (shadow_patch, offset) with padding."""
    pad = pad or blur * 2
    m = np.pad(patch[..., 3] / 255, pad)
    m = cv2.GaussianBlur(m, (0, 0), blur) * a
    return fill(m, (0, 0, 0)), (-pad, -pad + dy)


def card(w, h, r, color, a=1.0, stroke=None, stroke_w=2):
    p = fill(rrect(w, h, r), color, a)
    if stroke is not None:
        inner = rrect(w - 2 * stroke_w, h - 2 * stroke_w, max(r - stroke_w, 1))
        ring = rrect(w, h, r).copy()
        ring[stroke_w:h - stroke_w, stroke_w:w - stroke_w] -= inner
        over(p, fill(np.clip(ring, 0, 1), stroke[:3], stroke[3] if len(stroke) > 3 else 1))
    return p


def glass(base, x, y, w, h, r, blur=28, tint=(255, 255, 255), tint_a=0.10, edge_a=0.35):
    """Frosted glass patch: blurred copy of `base` (RGB float) under a rounded mask, plus tint and edge."""
    x, y = int(x), int(y)
    pad = blur * 2
    x0, y0, x1, y1 = max(x - pad, 0), max(y - pad, 0), min(x + w + pad, base.shape[1]), min(y + h + pad, base.shape[0])
    region = cv2.GaussianBlur(base[y0:y1, x0:x1], (0, 0), blur)
    sub = np.zeros((h, w, 3), np.float32)
    sx0, sy0 = x - x0, y - y0
    hh, ww = min(h, region.shape[0] - sy0), min(w, region.shape[1] - sx0)
    if hh > 0 and ww > 0:
        sub[:hh, :ww] = region[sy0:sy0 + hh, sx0:sx0 + ww]
    sub = sub * (1 - tint_a) + np.array(tint, np.float32) * tint_a
    m = rrect(w, h, r)[..., None]
    p = np.concatenate([sub * m, m * 255], -1)
    ring = rrect(w, h, r) - np.pad(rrect(w - 4, h - 4, max(r - 2, 1)), 2)
    over(p, fill(np.clip(ring, 0, 1), (255, 255, 255), edge_a))
    return p


# ---------- type ----------

@lru_cache(maxsize=128)
def font(key, size, wght=None):
    file, variable = FONTS[key]
    f = ImageFont.truetype(str(FONT_DIR / file), int(size))
    if variable and wght:
        f.set_variation_by_axes([wght])
    return f


@lru_cache(maxsize=1024)
def text_patch(text, key, size, color=(255, 255, 255), wght=None, tracking=0.0, pad=4, stroke=0,
               stroke_color=(0, 0, 0)):
    """Premultiplied RGBA patch of a line of text. tracking is in em; stroke is an outline in px.
    Returns (patch, baseline_y)."""
    f = font(key, size, wght)
    asc, desc = f.getmetrics()
    pad = pad + stroke
    if tracking:
        widths = [f.getlength(c) for c in text]
        total = sum(widths) + tracking * size * (len(text) - 1)
    else:
        total = f.getlength(text)
    w, h = int(math.ceil(total)) + 2 * pad + int(size * 0.2), asc + desc + 2 * pad

    def draw(sw):
        img = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(img)
        if tracking:
            x = pad
            for ch, cw in zip(text, widths):
                d.text((x, pad), ch, font=f, fill=255, stroke_width=sw, stroke_fill=255)
                x += cw + tracking * size
        else:
            d.text((pad, pad), text, font=f, fill=255, stroke_width=sw, stroke_fill=255)
        return img

    img = draw(0)
    m = np.asarray(img, np.float32) / 255
    if stroke:
        out = draw(stroke)
        bbox = out.getbbox() or (0, 0, 1, 1)
        p = fill(np.asarray(out, np.float32) / 255, stroke_color)
        over(p, fill(m, color))
    else:
        bbox = img.getbbox() or (0, 0, 1, 1)
        p = fill(m, color)
    return p[:, :bbox[2] + pad], asc + pad


def text_w(text, key, size, wght=None, tracking=0.0):
    f = font(key, size, wght)
    if tracking:
        return sum(f.getlength(c) for c in text) + tracking * size * (len(text) - 1)
    return f.getlength(text)


def fit_size(text, key, width, wght=None, tracking=0.0, lo=8, hi=900):
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if text_w(text, key, mid, wght, tracking) <= width:
            lo = mid
        else:
            hi = mid
    return lo


def put_text(dst, text, key, size, x, y, color=(255, 255, 255), wght=None, tracking=0.0, anchor="ls",
             opacity=1.0, scale=1.0, blur=0.0, stroke=0, stroke_color=(0, 0, 0)):
    """Draw text with its anchor at (x, y). anchor: l/m/r + s (baseline) or t/m (top/middle of patch)."""
    if size < 4:
        return 0
    p, base = text_patch(text, key, int(size), tuple(int(v) for v in color), wght, tracking, 4, int(stroke),
                         tuple(int(v) for v in stroke_color))
    if scale != 1.0:
        p = cv2.resize(p, (max(1, int(p.shape[1] * scale)), max(1, int(p.shape[0] * scale))),
                       interpolation=cv2.INTER_LINEAR)
        base *= scale
    if blur > 0.3:
        p = np.pad(p, ((int(blur * 3),) * 2, (int(blur * 3),) * 2, (0, 0)))
        base += int(blur * 3)
        p = cv2.GaussianBlur(p, (0, 0), blur)
    h, w = p.shape[:2]
    ox = {"l": 0, "m": w / 2, "r": w}[anchor[0]]
    oy = {"s": base, "t": 0, "m": h / 2}[anchor[1]]
    over(dst, p, x - ox, y - oy, opacity)
    return w


# ---------- strokes ----------

def polyline(dst, pts, color, width, progress=1.0, opacity=1.0):
    """Draw an anti-aliased polyline (canvas px), revealed along its length by progress (0..1)."""
    pts = np.asarray(pts, np.float32)
    if len(pts) < 2 or progress <= 0:
        return dst
    seg = np.hypot(*np.diff(pts, axis=0).T)
    total = seg.sum()
    keep, acc = [pts[0]], 0.0
    for p0, p1, s in zip(pts[:-1], pts[1:], seg):
        if acc + s >= progress * total:
            r = (progress * total - acc) / max(s, 1e-6)
            keep.append(p0 + (p1 - p0) * r)
            break
        keep.append(p1)
        acc += s
    k = np.array(keep)
    x0, y0 = np.floor(k.min(0) - width * 2).astype(int)
    x1, y1 = np.ceil(k.max(0) + width * 2).astype(int)
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    sh = 4  # sub-pixel precision for cv2
    cv2.polylines(m, [((k - [x0, y0]) * (1 << sh)).astype(np.int32)], False, 255, int(round(width)),
                  cv2.LINE_AA, shift=sh)
    over(dst, fill(m.astype(np.float32) / 255, color), x0, y0, opacity)
    return dst


# ---------- grade ----------

def grade(img, contrast=1.0, sat=1.0, warm=0.0, lift=0.0, gain=1.0, gamma=1.0):
    x = img / 255
    if gamma != 1.0:
        x = np.power(np.clip(x, 0, 1), 1 / gamma)
    x = (x - 0.5) * contrast + 0.5
    x = x * gain + lift
    if sat != 1.0:
        g = x @ np.array([0.299, 0.587, 0.114], np.float32)
        x = g[..., None] + (x - g[..., None]) * sat
    if warm:
        x = x + np.array([warm, warm * 0.25, -warm], np.float32)
    return np.clip(x, 0, 1) * 255


@lru_cache(maxsize=4)
def _vig(h, w, strength, roundness):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    d = np.hypot((xs / w - 0.5) * 2 * roundness, (ys / h - 0.5) * 2)
    return (1 - strength * np.clip(d - 0.35, 0, 1) ** 1.6)[..., None]


def vignette(img, strength=0.5, roundness=1.0):
    return img * _vig(img.shape[0], img.shape[1], strength, roundness)


@lru_cache(maxsize=1)
def _grain_bank(h, w):
    rng = np.random.default_rng(11)
    return [cv2.resize(rng.normal(0, 1, (h // 2, w // 2)).astype(np.float32), (w, h)) for _ in range(6)]


def grain(img, i, amount=6.0):
    g = _grain_bank(img.shape[0], img.shape[1])[i % 6]
    return img + g[..., None] * amount


def bloom(img, thresh=200, blur=30, amount=0.35):
    hi = np.clip(img - thresh, 0, None)
    return img + cv2.GaussianBlur(hi, (0, 0), blur) * amount


# ---------- output ----------

class Writer:
    """Pipes frames to ffmpeg. Silent by default (style showcases); audio=True keeps the clip's sound."""

    def __init__(self, out, clip, fps=None, crf=16, audio=False):
        self.out = str(out)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        src_audio = ["-i", clip.src, "-map", "0:v", "-map", "1:a?", "-c:a", "aac", "-b:a", "256k", "-shortest"] \
            if audio else ["-an"]
        self.p = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
             "-r", str(fps or clip.fps), "-i", "-", *src_audio,
             "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", self.out],
            stdin=subprocess.PIPE)

    def write(self, img):
        self.p.stdin.write(np.clip(img, 0, 255).astype(np.uint8).tobytes())

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def to_u8(img):
    return np.clip(img, 0, 255).astype(np.uint8)


def contact_sheet(frames, labels, out, cols=6, width=1800):
    tw = width // cols
    th = int(tw * H / W)
    rows = math.ceil(len(frames) / cols)
    sheet = np.zeros((rows * th, cols * tw, 3), np.uint8)
    for k, (f, lab) in enumerate(zip(frames, labels)):
        r, c = divmod(k, cols)
        t = cv2.resize(to_u8(f), (tw, th), interpolation=cv2.INTER_AREA)
        cv2.putText(t, lab, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
        sheet[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = t
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
