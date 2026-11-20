"""Premium motion layer for the v4 styles: springs, a smooth camera with motion blur, motion-blurred placement,
liquid glass (real refraction of whatever is underneath), liquid shape morphs, 3D cards, light and glow.
Everything is a pure function of time and the inputs, so renders stay frame-deterministic.

Conventions (same as core): canvas images are float32 RGB 0..255, layers/patches are premultiplied RGBA.
"""
import math
from functools import lru_cache

import cv2
import numpy as np

import core as c

W, H = c.W, c.H


# ---------- motion curves ----------

def spring(t, t0, freq=2.0, damp=0.45):
    """Damped spring 0 -> 1 from t0: fast, one soft overshoot, settles. freq in Hz, damp < 1."""
    x = t - t0
    if x <= 0:
        return 0.0
    w = 2 * math.pi * freq
    wd = w * math.sqrt(1 - damp * damp)
    return 1 - math.exp(-damp * w * x) * (math.cos(wd * x) + damp * w / wd * math.sin(wd * x))


def smooth(t, t0, dur):
    """Smootherstep 0 -> 1 over [t0, t0 + dur]."""
    x = c.clamp((t - t0) / max(dur, 1e-4))
    return x * x * x * (x * (x * 6 - 15) + 10)


def expo(t, t0, dur):
    return c.ease_out_expo(c.prog(t, t0, dur))


def vel(fn, t, fps=25.0, shutter=0.6):
    """Motion-blur vector (px) of a position function fn(t) -> (x, y) for a 'shutter' fraction of a frame."""
    a, b = fn(t - shutter / fps), fn(t)
    return b[0] - a[0], b[1] - a[1]


# ---------- blur ----------

def line_kernel(dx, dy):
    L = math.hypot(dx, dy)
    if L < 1.5:
        return None
    n = int(math.ceil(L)) | 1
    k = np.zeros((n, n), np.float32)
    cx = n / 2 - 0.5
    ux, uy = dx / L, dy / L
    cv2.line(k, (int(round((cx - ux * L / 2) * 16)), int(round((cx - uy * L / 2) * 16))),
             (int(round((cx + ux * L / 2) * 16)), int(round((cx + uy * L / 2) * 16))), 1.0, 1, cv2.LINE_AA, 4)
    s = k.sum()
    return k / s if s > 0 else None


def blur_patch(patch, dx, dy):
    """Directional motion blur of a premultiplied patch. Returns (patch, pad): the patch grows by pad per side."""
    k = line_kernel(dx, dy)
    if k is None:
        return patch, 0
    pad = k.shape[0] // 2 + 1
    p = np.pad(patch, ((pad, pad), (pad, pad), (0, 0)))
    return cv2.filter2D(p, -1, k, borderType=cv2.BORDER_CONSTANT), pad


def dir_blur(img, dx, dy):
    """Directional blur of a whole image (whip / slide smear)."""
    k = line_kernel(dx, dy)
    return img if k is None else cv2.filter2D(img, -1, k, borderType=cv2.BORDER_REFLECT)


def zoom_blur(img, cx, cy, amount, n=7):
    """Radial zoom smear about (cx, cy); amount is the relative scale spread (0.03 = a fast push)."""
    if abs(amount) < 0.004:
        return img
    acc = np.zeros_like(img)
    h, w = img.shape[:2]
    for k in range(n):
        s = 1 + amount * (k / (n - 1) - 0.5)
        M = np.float32([[s, 0, cx - s * cx], [0, s, cy - s * cy]])
        acc += cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    return acc / n


# ---------- camera ----------

class Cam:
    """Smooth virtual camera over a core.Framing.

    keys: [{"t": 0, "zoom": 1.0}, {"t": 0.4, "zoom": 1.15, "dx": 0, "dy": -40, "rot": 0, "dur": 0.5,
            "ease": "expo" | "smooth" | "spring" | "cut", "freq": 2, "damp": .5}, ...]
    Each key eases from wherever the camera is to its targets. Zoom is about the face. Moves get real motion blur
    (radial for zoom, directional for pans). drift adds a slow floating hand-held feel (px)."""

    P = ("zoom", "dx", "dy", "rot")

    def __init__(self, fr, keys, drift=0.0, blur=1.0):
        self.fr = fr
        self.keys = sorted(keys, key=lambda k: k["t"])
        self.drift = drift
        self.blur = blur

    def at(self, t):
        cur = {"zoom": 1.0, "dx": 0.0, "dy": 0.0, "rot": 0.0}
        cur.update({p: self.keys[0][p] for p in self.P if p in self.keys[0]})
        for k in self.keys[1:]:
            if t < k["t"]:
                break
            mode, d = k.get("ease", "expo"), k.get("dur", 0.5)
            if mode == "cut":
                e = 1.0
            elif mode == "spring":
                e = spring(t, k["t"], k.get("freq", 2.0), k.get("damp", 0.5))
            elif mode == "smooth":
                e = smooth(t, k["t"], d)
            else:
                e = expo(t, k["t"], d)
            cur = {p: cur[p] + (k.get(p, cur[p]) - cur[p]) * e for p in cur}
        if self.drift:
            a = self.drift
            cur["dx"] += a * (math.sin(t * 1.7 + 0.4) + 0.5 * math.sin(t * 3.1 + 2.0))
            cur["dy"] += a * 0.7 * (math.sin(t * 1.3 + 1.1) + 0.4 * math.sin(t * 2.7))
            cur["rot"] += a * 0.012 * math.sin(t * 1.1 + 0.7)
        return cur

    def place(self, i, t, rgb, alpha):
        p = self.at(t)
        fr = self.fr
        fb = fr.clip.face[min(i, fr.clip.n - 1)]
        zc = (fb[0] - fr.x0, fb[1] - fr.y0)
        img, a = fr.place_t(rgb, alpha, p["zoom"], zc, p["dx"], p["dy"], p["rot"])
        if not self.blur:
            return img, a
        q = self.at(t - 0.6 / fr.clip.fps)
        dz = (p["zoom"] - q["zoom"]) / p["zoom"] * self.blur
        mx, my = (p["dx"] - q["dx"]) * self.blur, (p["dy"] - q["dy"]) * self.blur
        if abs(dz) < 0.004 and math.hypot(mx, my) < 1.5:
            return img, a
        st = np.concatenate([img, a[..., None] * 255], -1)
        if abs(dz) >= 0.004:
            fcx, fcy = fr.face_box(i)[:2]
            st = zoom_blur(st, fcx, fcy, dz * 2.2)
        st = dir_blur(st, mx, my)
        return st[..., :3], st[..., 3] / 255


# ---------- placement ----------

def put(dst, patch, cx, cy, scale=1.0, rot=0.0, opacity=1.0, mv=(0.0, 0.0), shadow=0.0, sblur=26, sdy=16):
    """Place a premultiplied patch centred on (cx, cy), scaled and rotated (deg), motion-blurred along mv (px)."""
    if opacity <= 0.003 or scale <= 0.01:
        return dst
    p = patch
    if scale != 1.0 or rot:
        h, w = p.shape[:2]
        r = math.radians(rot)
        nw = int(math.ceil((abs(w * math.cos(r)) + abs(h * math.sin(r))) * scale)) + 4
        nh = int(math.ceil((abs(w * math.sin(r)) + abs(h * math.cos(r))) * scale)) + 4
        M = cv2.getRotationMatrix2D((w / 2, h / 2), rot, scale)
        M[0, 2] += nw / 2 - w / 2
        M[1, 2] += nh / 2 - h / 2
        p = cv2.warpAffine(p, M, (nw, nh), flags=cv2.INTER_LINEAR, borderValue=0)
    p, _ = blur_patch(p, *mv)
    h, w = p.shape[:2]
    if shadow > 0:
        sh, (ox, oy) = c.shadow(p, sblur, shadow, sdy)
        c.over(dst, sh, cx - w / 2 + ox, cy - h / 2 + oy, opacity)
    c.over(dst, p, cx - w / 2, cy - h / 2, opacity)
    return dst


def add(dst, patch, x, y, amount=1.0):
    """Additive light: add a premultiplied patch's colour onto dst (RGB or RGBA) at (x, y)."""
    h, w = patch.shape[:2]
    x, y = int(round(x)), int(round(y))
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, dst.shape[1]), min(y + h, dst.shape[0])
    if x0 >= x1 or y0 >= y1:
        return dst
    s = patch[y0 - y:y1 - y, x0 - x:x1 - x]
    dst[y0:y1, x0:x1, :3] += s[..., :3] * amount
    if dst.shape[2] == 4:
        dst[y0:y1, x0:x1, 3] = np.minimum(255, dst[y0:y1, x0:x1, 3] + s[..., 3] * amount)
    return dst


def text(s, key="grotesk", size=80, color=(255, 255, 255), wght=None, tracking=0.0, stroke=0, stroke_color=(0, 0, 0)):
    return c.text_patch(s, key, size, color, wght, tracking, 4, stroke, stroke_color)[0]


def glow(patch, sigma=18, amount=0.8, pad=None):
    """Patch plus a soft bloom of itself (neon / light text). Grows by pad per side."""
    pad = int(sigma * 2.5) if pad is None else pad
    p = np.pad(patch, ((pad, pad), (pad, pad), (0, 0)))
    g = cv2.GaussianBlur(p, (0, 0), sigma)
    out = p + g * amount
    out[..., 3] = np.minimum(out[..., 3], 255)
    return out


def light_sweep(patch, p, width=0.16, angle=22, strength=0.55):
    """Specular band sweeping across a patch as p goes 0 -> 1 (clipped to the patch alpha)."""
    if p <= 0 or p >= 1:
        return patch
    h, w = patch.shape[:2]
    ys, xs = grid(h, w)
    a = math.radians(angle)
    u = (xs * math.cos(a) + ys * math.sin(a)) / (w * math.cos(a) + h * math.sin(a))
    band = np.exp(-((u - (p * 1.4 - 0.2)) / width) ** 2) * strength
    out = patch.copy()
    out[..., :3] += band[..., None] * patch[..., 3:4]
    return out


def gradient(h, w, c0, c1, angle=90):
    """Linear gradient image (RGB float) from c0 to c1; angle 90 = top to bottom."""
    ys, xs = grid(h, w)
    a = math.radians(angle)
    u = xs * math.cos(a) + ys * math.sin(a)
    u = (u - u.min()) / max(u.max() - u.min(), 1e-6)
    return np.array(c0, np.float32) * (1 - u[..., None]) + np.array(c1, np.float32) * u[..., None]


def rgb_shift(img, px, angle=0.0):
    """Chromatic split: red one way, blue the other."""
    if abs(px) < 0.5:
        return img
    dx, dy = px * math.cos(math.radians(angle)), px * math.sin(math.radians(angle))
    h, w = img.shape[:2]
    out = img.copy()
    for ch, s in ((0, 1), (2, -1)):
        M = np.float32([[1, 0, dx * s], [0, 1, dy * s]])
        out[..., ch] = cv2.warpAffine(img[..., ch], M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return out


def flare(dst, x, y, length=900, thick=10, color=(120, 190, 255), amount=1.0):
    """Anamorphic lens flare: a thin horizontal streak with a hot core, added as light."""
    if amount <= 0.01:
        return dst
    hw, hh = int(length / 2), int(thick * 6)
    ys, xs = grid(2 * hh, 2 * hw)
    u, v = (xs - hw) / hw, (ys - hh) / thick
    streak = np.exp(-v * v) * np.exp(-np.abs(u) * 3.2)
    core = np.exp(-((xs - hw) ** 2 + (ys - hh) ** 2) / (2 * (thick * 2.2) ** 2))
    m = streak * 0.9 + core
    p = np.concatenate([m[..., None] * np.array(color, np.float32), m[..., None] * 255], -1)
    p[..., :3] += core[..., None] * 160
    return add(dst, p, x - hw, y - hh, amount)


# ---------- shapes ----------

@lru_cache(maxsize=16)
def grid(h, w):
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    ys.setflags(write=False)
    xs.setflags(write=False)
    return ys, xs


def sdf_pill(xs, ys, cx, cy, w, h, r=None):
    """Signed distance to a rounded rectangle (r = None -> full pill)."""
    r = min(w, h) / 2 if r is None else min(r, w / 2, h / 2)
    qx = np.abs(xs - cx) - (w / 2 - r)
    qy = np.abs(ys - cy) - (h / 2 - r)
    return np.hypot(np.maximum(qx, 0), np.maximum(qy, 0)) + np.minimum(np.maximum(qx, qy), 0) - r


def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)


def liquid(w, h, shapes, k=40):
    """Mask (h, w) of rounded rects / pills / circles melted together like liquid.
    shapes: [(cx, cy, sw, sh[, r]), ...] in patch pixels; k = how far they melt into each other."""
    ys, xs = grid(h, w)
    d = None
    for s in shapes:
        if s[2] <= 1 or s[3] <= 1:
            continue
        di = sdf_pill(xs, ys, *s)
        d = di if d is None else smin(d, di, k)
    if d is None:
        return np.zeros((h, w), np.float32)
    return np.clip(0.5 - d, 0, 1)


# ---------- liquid glass ----------

def glass(base, mask, x, y, bevel=None, refract=1.0, magnify=0.06, frost=2.5, chroma=0.5, tint=(255, 255, 255),
          tint_a=0.06, light=(-0.55, -0.83), spec=0.85, rim=0.75, depth=0.12):
    """Liquid glass over `base` (RGB float canvas). mask (h, w) in [0..1] sits at canvas (x, y).

    What's underneath is refracted through a convex bevel: the rim pulls in content from just outside the shape
    (the liquid-glass lensing), the centre magnifies slightly, colour splits at the rim (chroma), a light from the
    top-left lays a specular arc on the rim, a thin bright edge outlines the shape and a faint gloss sits on top.
    Returns a premultiplied patch (h, w, 4) to over() at (x, y)."""
    h, w = mask.shape
    x, y = int(round(x)), int(round(y))
    bevel = float(bevel or max(10.0, min(h, w) * 0.3))
    m8 = (mask > 0.5).astype(np.uint8)
    d = cv2.distanceTransform(np.pad(m8, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    u = np.clip(d / bevel, 0, 1)
    hgt = cv2.GaussianBlur(1 - (1 - u) ** 2.2, (0, 0), 1.4)
    gy, gx = np.gradient(hgt)
    gx, gy = gx * bevel, gy * bevel  # slope, ~0..2 at the rim
    ys, xs = grid(h, w)
    pad = int(bevel * refract * 1.6) + 8
    X0, Y0 = x - pad, y - pad
    bh, bw = base.shape[:2]
    sx0, sy0, sx1, sy1 = max(X0, 0), max(Y0, 0), min(x + w + pad, bw), min(y + h + pad, bh)
    if sx0 >= sx1 or sy0 >= sy1:
        return np.zeros((h, w, 4), np.float32)
    src = np.zeros((h + 2 * pad, w + 2 * pad, 3), np.float32)
    src[sy0 - Y0:sy1 - Y0, sx0 - X0:sx1 - X0] = base[sy0:sy1, sx0:sx1]
    if sx0 > X0 or sy0 > Y0 or sx1 < x + w + pad or sy1 < y + h + pad:  # off-canvas: extend the edge
        src = cv2.copyMakeBorder(base[sy0:sy1, sx0:sx1], sy0 - Y0, (y + h + pad) - sy1, sx0 - X0,
                                 (x + w + pad) - sx1, cv2.BORDER_REPLICATE)
    if frost > 0:
        src = cv2.GaussianBlur(src, (0, 0), frost)
    k = refract * bevel * 0.55
    mx, my = (w / 2 - xs) * magnify, (h / 2 - ys) * magnify
    out = np.zeros((h, w, 3), np.float32)
    for ch, cs in ((0, 1 - chroma * 0.35), (1, 1.0), (2, 1 + chroma * 0.35)):
        mapx = xs + pad - gx * k * cs + mx
        mapy = ys + pad - gy * k * cs + my
        out[..., ch] = cv2.remap(src[..., ch], mapx, mapy, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    n = np.hypot(gx, gy) + 1e-6
    lx, ly = light
    facing = -(gx * lx + gy * ly) / n
    edge = (1 - u) ** 1.6
    sp = (np.clip(facing, 0, 1) ** 3 * edge * spec + np.clip(-facing, 0, 1) ** 3 * edge * spec * 0.35)
    line = np.exp(-(d / 1.3) ** 2) * rim
    gloss = np.clip(1 - ys / (h * 0.55), 0, 1) ** 2 * 0.07 * u
    out = out * (1 - tint_a) + np.array(tint, np.float32) * tint_a
    out = out * (1 - depth * edge[..., None] * np.clip(-facing, 0, 1)[..., None])  # darker far rim
    out += (sp + line + gloss)[..., None] * 255
    a = mask[..., None]
    return np.concatenate([np.clip(out, 0, 255) * a, a * 255], -1)


def glass_shadow(dst, mask, x, y, blur=30, a=0.35, dy=18):
    """Soft contact shadow under a glass shape (dst RGB canvas)."""
    pad = int(blur * 2.5)
    m = cv2.GaussianBlur(np.pad(mask, pad), (0, 0), blur) * a
    region = dst[max(y + dy - pad, 0):, max(x - pad, 0):]
    yy, xx = max(y + dy - pad, 0) - (y + dy - pad), max(x - pad, 0) - (x - pad)
    hh, ww = min(m.shape[0] - yy, region.shape[0]), min(m.shape[1] - xx, region.shape[1])
    if hh > 0 and ww > 0:
        region[:hh, :ww] *= (1 - m[yy:yy + hh, xx:xx + ww])[..., None]
    return dst


# ---------- 3D ----------

def rot3(rx, ry, rz):
    a, b, g = (math.radians(v) for v in (rx, ry, rz))
    Rx = np.array([[1, 0, 0], [0, math.cos(a), -math.sin(a)], [0, math.sin(a), math.cos(a)]], np.float32)
    Ry = np.array([[math.cos(b), 0, math.sin(b)], [0, 1, 0], [-math.sin(b), 0, math.cos(b)]], np.float32)
    Rz = np.array([[math.cos(g), -math.sin(g), 0], [math.sin(g), math.cos(g), 0], [0, 0, 1]], np.float32)
    return Rz @ Ry @ Rx


def project(pts, cx, cy, rx=0.0, ry=0.0, rz=0.0, scale=1.0, z=0.0, persp=1800.0):
    """Rotate 3D points (N, 3) about the origin, push back by z, project to the canvas around (cx, cy)."""
    P = (np.asarray(pts, np.float32) * scale) @ rot3(rx, ry, rz).T
    zz = P[:, 2] + persp + z
    return P[:, :2] * (persp / zz)[:, None] + np.float32([cx, cy]), zz


def card3d(dst, patch, cx, cy, rx=0.0, ry=0.0, rz=0.0, scale=1.0, z=0.0, persp=1800.0, opacity=1.0, shade=0.35,
           shadow=0.0, back=None):
    """Draw a flat patch as a card rotated in 3D (deg) with perspective. Shades as it turns away from the light;
    when it faces backwards draws `back` instead (if given). Returns the projected quad (4, 2)."""
    h, w = patch.shape[:2]
    corners = np.float32([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0], [w / 2, h / 2, 0], [-w / 2, h / 2, 0]])
    quad, _ = project(corners, cx, cy, rx, ry, rz, scale, z, persp)
    nrm = rot3(rx, ry, rz) @ np.float32([0, 0, -1])
    facing = float(-nrm[2])  # 1 = facing the camera
    src = patch
    if facing < 0:
        if back is None:
            return quad
        src = back[:, ::-1].copy()
    hh, ww = src.shape[:2]
    x0, y0 = np.floor(quad.min(0)).astype(int) - 2
    x1, y1 = np.ceil(quad.max(0)).astype(int) + 2
    x0, y0 = max(x0, -4000), max(y0, -4000)
    if x1 - x0 < 2 or y1 - y0 < 2 or x1 - x0 > 6000 or y1 - y0 > 6000:
        return quad
    sq = np.float32([[0, 0], [ww, 0], [ww, hh], [0, hh]])
    M = cv2.getPerspectiveTransform(sq, (quad - np.float32([x0, y0])).astype(np.float32))
    out = cv2.warpPerspective(src, M, (int(x1 - x0), int(y1 - y0)), flags=cv2.INTER_LINEAR, borderValue=0)
    if shade:
        out[..., :3] *= 1 - shade * (1 - abs(facing))
    if shadow > 0:
        sh, (ox, oy) = c.shadow(out, 34, shadow, 26)
        c.over(dst, sh, x0 + ox, y0 + oy, opacity)
    c.over(dst, out, x0, y0, opacity)
    return quad


def reflection(patch, length=0.45, a=0.28):
    """Faded mirror of a patch for a glossy floor (place it right under the patch)."""
    h = patch.shape[0]
    n = max(1, int(h * length))
    r = patch[::-1][:n].copy()
    r *= (np.linspace(1, 0, n, dtype=np.float32) ** 1.5 * a)[:, None, None]
    return r


# ---------- matte helpers ----------

def contour(alpha, thresh=0.5, smooth_px=6):
    """Largest outline of the person matte as (N, 2) canvas points (for neon traces and outlines)."""
    m = (alpha > thresh).astype(np.uint8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cs:
        return np.zeros((0, 2), np.float32)
    pts = max(cs, key=len)[:, 0].astype(np.float32)
    if smooth_px and len(pts) > 4 * smooth_px:
        k = np.ones(2 * smooth_px + 1, np.float32) / (2 * smooth_px + 1)
        pts = np.stack([np.convolve(np.r_[pts[-smooth_px:, j], pts[:, j], pts[:smooth_px, j]], k, "valid")
                        for j in range(2)], 1)
    return pts


def outline(alpha, width=10, soft=1.5):
    """Ring mask around the person (sticker / neon edge)."""
    m = (alpha > 0.5).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1))
    ring = cv2.dilate(m, k).astype(np.float32) - m
    return cv2.GaussianBlur(ring, (0, 0), soft) if soft else ring
