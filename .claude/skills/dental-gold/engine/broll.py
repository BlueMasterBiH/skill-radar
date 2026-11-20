"""Full-screen drawn b-roll for dental-gold (dynamic cut). Black and gold, drawn in code, no stock.

Each class draws a whole 1080x1920 frame for t in [t0, t1]; dynamic.py handles the transitions in and out and
puts the speaker in a picture-in-picture window. `marks` are times (already resolved) for the beats inside.

kinds:
  xray      a dental X-ray; a gold scan line reveals it, then rings mark the tooth, the root and the bone
            marks: zub, korijen, kost
  root      the root canal animation, large            marks: remove, clean, fill, keep
  restore   implant, crown and bridge pop in, then a smile sparkle   marks: implant, krunica, most, osmijeh
  pain      a tooth throbbing with gold pain lines
  versus    two teeth: treat the nerve or pull the tooth           marks: left, right
  extract   a tooth lifts out of the gum and leaves a dashed gap   marks: out
"""
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

import elements as E

W, H = 1080, 1920
GOLD, GOLD_HI, WHITE, IVORY, GUM = E.AQUA, E.GOLD_HI, E.WHITE, E.IVORY, E.GUM
SS = 2


def anton(size):
    return E.font(size, "Regular", "Anton-Regular.ttf")


_bg = {}


def background(seed=0):
    """warm black gradient with a faint gold dot grid and a soft gold glow"""
    if seed not in _bg:
        y = np.linspace(0, 1, H)[:, None, None]
        top, bot = np.array([10, 9, 7], np.float32), np.array([30, 25, 17], np.float32)
        img = (top * (1 - y) + bot * y) * np.ones((1, W, 1), np.float32)
        yy, xx = np.mgrid[0:H, 0:W]
        g = np.exp(-(((xx - 540) / 620.0) ** 2 + ((yy - 900) / 760.0) ** 2))
        img += g[..., None] * np.array([46, 36, 16], np.float32)
        dots = ((xx % 54 == 27) & (yy % 54 == 27)).astype(np.float32)
        dots = cv2.GaussianBlur(dots, (0, 0), 1.0) * 6
        img += dots[..., None] * np.array([200, 160, 80], np.float32) * 0.25
        _bg[seed] = img
    return _bg[seed].copy()


def push(img, t, t0, t1, amount=0.06):
    """slow push-in over the shot so the frame never sits still"""
    u = E.clamp((t - t0) / max(0.1, t1 - t0))
    z = 1 + amount * u
    M = np.float32([[z, 0, 540 - z * 540], [0, z, 900 - z * 900]])
    return cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)


def kicker(img, text, t, t0):
    """small gold heading top-left with a drawing underline"""
    a = E.clamp((t - t0 - 0.1) / 0.25)
    if a <= 0:
        return
    f = E.font(52, "Bold")
    im = Image.new("RGBA", (900, 90), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.text((0, 0), text, font=f, fill=GOLD + (255,))
    lw = int((E.text_w(text, f)) * E.ease_out(t, t0 + 0.15, 0.5))
    d.rectangle((0, 72, lw, 77), fill=GOLD + (255,))
    E.put(img, im, 72 + 450, 250 + 45 + (1 - a) * 16, 1.0, a)


def label(text, size=58, color=WHITE, wt="Bold"):
    f = E.font(size, wt)
    w = int(E.text_w(text, f)) + 40
    im = Image.new("RGBA", (w, size + 40), (0, 0, 0, 0))
    ImageDraw.Draw(im).text((20, 10), text, font=f, fill=color + (255,))
    return E.with_shadow(im, 14, 0.6, 4)


def ring(d, x, y, r, prog, width=6):
    if prog <= 0:
        return
    d.arc((x - r, y - r, x + r, y + r), -90, -90 + 360 * prog, fill=GOLD_HI + (255,), width=width)


# ---------- tooth outlines (unit coords, y down, about 1 x 1) ----------
MOLAR = E.smooth_poly([(-0.48, -0.42), (-0.30, -0.50), (-0.12, -0.44), (0.0, -0.48), (0.14, -0.44), (0.32, -0.50),
                       (0.48, -0.40), (0.46, -0.12), (0.36, 0.02), (0.30, 0.48), (0.18, 0.50), (0.12, 0.10),
                       (-0.10, 0.10), (-0.18, 0.50), (-0.32, 0.48), (-0.38, 0.02), (-0.47, -0.12)], 3)


def tooth_pts(poly, cx, cy, w, h, s=1):
    return [(s * (cx + x * w), s * (cy + y * h)) for x, y in poly]


# ---------- kinds ----------
class XRay:
    def __init__(self, t0, t1, marks, title="SNIMAK"):
        self.t0, self.t1, self.m, self.title = t0, t1, marks, title
        self.base = self._film()

    def _film(self):
        """grey X-ray film: bone, three teeth with canals, drawn once"""
        w, h = 900, 900
        rng = np.random.default_rng(3)
        bone = cv2.GaussianBlur(rng.random((h, w)).astype(np.float32), (0, 0), 6) * 60 + 60
        yy = np.linspace(0, 1, h)[:, None]
        bone *= (0.35 + 0.65 * np.clip((yy - 0.38) / 0.2, 0, 1))      # bone only below the gum line
        im = Image.fromarray(np.clip(bone, 0, 255).astype(np.uint8)).convert("L")
        d = ImageDraw.Draw(im)
        for k, (cx, sc) in enumerate([(170, 0.85), (450, 1.0), (730, 0.85)]):
            tw, th = 250 * sc, 560 * sc
            cy = 430
            d.polygon(tooth_pts(E.RC_TOOTH, cx, cy, tw, th), fill=205)
            d.polygon(tooth_pts(E.smooth_poly([(-0.42, -0.30), (-0.34, -0.45), (0, -0.42), (0.34, -0.45), (0.42, -0.30),
                                               (0.36, -0.18), (-0.36, -0.18)], 3), cx, cy, tw, th), fill=240)
            d.polygon(tooth_pts(E.RC_PULP, cx, cy, tw, th), fill=95)
        im = im.filter(ImageFilter.GaussianBlur(2.2))
        a = np.asarray(im, np.float32)
        a += rng.normal(0, 5, a.shape)
        rgb = np.stack([a * 0.92, a * 0.96, a * 1.0], -1)          # cold film tint
        r = np.hypot(*np.mgrid[-1:1:complex(h), -1:1:complex(w)])
        rgb *= np.clip(1.25 - 0.55 * r, 0, 1)[..., None]
        return np.clip(rgb, 0, 255)

    def draw(self, t):
        img = background()
        kicker(img, self.title, t, self.t0)
        film = self.base.copy()
        # gold scan line reveals the film top to bottom
        u = E.ease_out(t, self.t0 + 0.05, 0.9)
        yline = int(900 * u)
        film[yline:] *= 0.08
        if u < 1:
            film[max(0, yline - 6):yline + 2] = np.array(GOLD_HI, np.float32)
            glow = np.exp(-((np.arange(900) - yline) / 40.0) ** 2)[:, None, None]
            film += glow * np.array([90, 70, 30], np.float32)
        x0, y0 = 90, 470
        img[y0:y0 + 900, x0:x0 + 900] = np.clip(film, 0, 255) * 0.97 + img[y0:y0 + 900, x0:x0 + 900] * 0.03
        cv2.rectangle(img, (x0 - 2, y0 - 2), (x0 + 901, y0 + 901), tuple(int(c) for c in GOLD), 2)
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        cx = x0 + 450
        spots = [("zub", "ZUB", cx, y0 + 250, 120, (cx + 175, y0 + 120)),
                 ("korijen", "KORIJEN", cx, y0 + 600, 95, (cx + 150, y0 + 690)),
                 ("kost", "KOST", x0 + 300, y0 + 760, 80, (x0 + 40, y0 + 930))]
        for key, txt, x, y, r, (lx, ly) in spots:
            tm = self.m.get(key)
            if tm is None or t < tm - 0.05:
                continue
            pr = E.ease_out(t, tm - 0.05, 0.45)
            ring(d, x, y, r, pr)
            if pr > 0.5:
                d.line((x + r * 0.7, y + (r * 0.7 if ly > y else -r * 0.7), lx, ly), fill=GOLD_HI + (255,), width=4)
        img = E_composite(img, ov)
        for key, txt, x, y, r, (lx, ly) in spots:
            tm = self.m.get(key)
            if tm is None or t < tm + 0.15:
                continue
            a = E.clamp((t - tm - 0.15) / 0.2)
            E.put(img, label(txt, 60, GOLD_HI), lx + 70, ly + (30 if ly > y else -30), 0.9 + 0.1 * a, a)
        return push(img, t, self.t0, self.t1, 0.05)


def E_composite(img, ov):
    a = np.asarray(ov, np.float32)
    al = a[..., 3:4] / 255
    return img * (1 - al) + a[..., :3] * al


class Root:
    def __init__(self, t0, t1, marks, title="Liječenje kanala"):
        self.t0, self.t1 = t0, t1
        self.el = E.RootCanal(t0 - 1.0, t1 + 1.0, title, marks.get("remove"), marks.get("clean"), marks.get("fill"),
                              marks.get("keep"), cy=930, zoom=1.38)
        self.el.t0 = t0 - 0.6                                     # pop already finished when the b-roll lands

    def draw(self, t):
        img = background()
        self.el.draw(img, t)
        return push(img, t, self.t0, self.t1, 0.04)


def implant_icon(d, cx, cy, s, prog):
    """screw in gold with an ivory crown on top"""
    top = cy - 0.15 * s
    for k in range(6):
        y = top + 0.1 * s + k * 0.085 * s
        wv = 0.17 * s * (1 - k * 0.09)
        d.rounded_rectangle((cx - wv, y, cx + wv, y + 0.05 * s), int(0.02 * s), fill=GOLD + (255,))
    d.polygon([(cx - 0.1 * s, top + 0.6 * s), (cx + 0.1 * s, top + 0.6 * s), (cx, top + 0.72 * s)], fill=GOLD + (255,))
    d.rectangle((cx - 0.07 * s, top - 0.05 * s, cx + 0.07 * s, top + 0.1 * s), fill=GOLD_HI + (255,))
    off = (1 - prog) * -0.15 * s
    al = int(255 * E.clamp(prog * 2))
    d.polygon(tooth_pts(E.smooth_poly([(-0.5, 0.0), (-0.4, -0.45), (0, -0.38), (0.4, -0.45), (0.5, 0.0), (0.38, 0.25),
                                       (-0.38, 0.25)], 3), cx, top - 0.2 * s + off, 0.42 * s, 0.4 * s),
              fill=IVORY + (al,), outline=GOLD + (al,), width=3)


def crown_icon(d, cx, cy, s, prog):
    d.rounded_rectangle((cx - 0.16 * s, cy - 0.05 * s, cx + 0.16 * s, cy + 0.4 * s), int(0.05 * s), fill=(200, 160, 120, 255))
    off = (1 - prog) * -0.15 * s
    al = int(255 * E.clamp(prog * 2))
    d.polygon(tooth_pts(E.smooth_poly([(-0.5, 0.25), (-0.46, -0.3), (-0.2, -0.48), (0, -0.4), (0.2, -0.48), (0.46, -0.3),
                                       (0.5, 0.25)], 3), cx, cy - 0.05 * s + off, 0.46 * s, 0.5 * s),
              fill=IVORY + (al,), outline=GOLD + (al,), width=3)


def bridge_icon(d, cx, cy, s, prog):
    d.rounded_rectangle((cx - 0.6 * s, cy + 0.14 * s, cx + 0.6 * s, cy + 0.42 * s), int(0.12 * s), fill=GUM + (255,))
    off = (1 - prog) * -0.3 * s
    for k, dx in enumerate((-0.38, 0.0, 0.38)):
        d.polygon(tooth_pts(E.smooth_poly([(-0.5, 0.3), (-0.45, -0.35), (0, -0.45), (0.45, -0.35), (0.5, 0.3)], 3),
                            cx + dx * s, cy + off, 0.34 * s, 0.5 * s), fill=IVORY + (255,), outline=GOLD + (255,), width=3)
    d.line((cx - 0.38 * s, cy + 0.02 * s + off, cx + 0.38 * s, cy + 0.02 * s + off), fill=GOLD_HI + (255,), width=5)


class Restore:
    def __init__(self, t0, t1, marks, title="NADOKNADA ZUBA"):
        self.t0, self.t1, self.m, self.title = t0, t1, marks, title

    def draw(self, t):
        img = background()
        kicker(img, self.title, t, self.t0)
        items = [("implant", "Implantat", implant_icon, 300), ("krunica", "Krunice", crown_icon, 1010),
                 ("most", "Mostovi", bridge_icon, 1430)]
        times = [self.m.get(k, self.t0 + 0.3 + 0.6 * i) for i, (k, *_r) in enumerate(items)]
        ov = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        rows = [(540, 600), (540, 1000), (540, 1400)]
        for (key, txt, fn, _), tm, (cx, cy) in zip(items, times, rows):
            if t < tm - 0.05:
                continue
            pr = E.ease_out(t, tm - 0.05, 0.5)
            fn(d, SS * (cx - 220), SS * cy, SS * 420, pr)
        ov = ov.resize((W, H), Image.LANCZOS)
        img = E_composite(img, ov)
        for (key, txt, fn, _), tm, (cx, cy) in zip(items, times, rows):
            if t < tm + 0.1:
                continue
            a = E.clamp((t - tm - 0.1) / 0.2)
            im = label(txt, 70)
            E.put(img, im, 560 + im.width / 2 + (1 - a) * 30, cy, 1.0, a)
        tw = self.m.get("osmijeh")
        if tw is not None and t >= tw:
            g = E.clamp((t - tw) / 0.8)
            sp = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            sd = ImageDraw.Draw(sp)
            for (x, y, r, dl) in ((300, 520, 36, 0), (820, 960, 28, 0.15), (380, 1330, 30, 0.3)):
                E.sparkle(sd, x, y, r * math.sin(math.pi * E.clamp(g * 1.4 - dl)), 1)
            img = E_composite(img, sp)
        return push(img, t, self.t0, self.t1, 0.05)


class Pain:
    def __init__(self, t0, t1, marks, title=""):
        self.t0, self.t1 = t0, t1

    def draw(self, t):
        img = background()
        k = t - self.t0
        throb = 0.5 + 0.5 * math.sin(k * 11)
        ov = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        cx, cy, s = 540, 900, 1 + 0.04 * throb
        d.polygon(tooth_pts(MOLAR, cx, cy, 560 * s, 640 * s, SS), fill=IVORY + (255,), outline=GOLD + (255,), width=6)
        d.polygon(tooth_pts(E.RC_PULP, cx, cy + 40, 400 * s, 560 * s, SS), fill=E.mix(E.PULP, (205, 95, 105), throb) + (255,))
        for j in range(8):                                     # jagged pain lines
            ang = j / 8 * 2 * math.pi + 0.3
            r0, r1 = 420 + 30 * throb, 560 + 50 * throb
            pts = []
            for q in range(5):
                r = r0 + (r1 - r0) * q / 4
                side = 18 if q % 2 else -18
                pts.append((SS * (cx + r * math.cos(ang) + side * math.sin(ang)),
                            SS * (cy + r * math.sin(ang) - side * math.cos(ang))))
            d.line(pts, fill=GOLD_HI + (int(160 + 95 * throb),), width=SS * 9, joint="curve")
        ov = ov.resize((W, H), Image.LANCZOS)
        img = E_composite(img, ov)
        sh = 10 * throb * (1 - E.clamp(k / 1.5))
        M = np.float32([[1, 0, sh * math.sin(k * 40)], [0, 1, sh * math.cos(k * 33)]])
        return cv2.warpAffine(push(img, t, self.t0, self.t1, 0.08), M, (W, H), borderMode=cv2.BORDER_REFLECT)


class Versus:
    def __init__(self, t0, t1, marks, title=""):
        self.t0, self.t1, self.m = t0, t1, marks

    def draw(self, t):
        img = background()
        tl, tr = self.m.get("left", self.t0 + 0.1), self.m.get("right", self.t0 + 1.0)
        ov = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        pl, pr = E.ease_out(t, tl - 0.05, 0.45), E.ease_out(t, tr - 0.05, 0.45)
        if t >= tl - 0.05:
            cx, cy = 290, 900 + (1 - pl) * 80
            d.polygon(tooth_pts(E.RC_TOOTH, cx, cy, 330, 560, SS), fill=IVORY + (int(255 * pl),), outline=GOLD + (int(255 * pl),), width=5)
            d.polygon(tooth_pts(E.RC_PULP, cx, cy, 330, 560, SS), fill=GOLD + (int(255 * pl),))
        if t >= tr - 0.05:
            lift = E.ease_out(t, tr + 0.1, 0.6) * 160
            cx, cy = 790, 900 + (1 - pr) * 80
            d.rounded_rectangle((SS * (cx - 220), SS * (cy + 60), SS * (cx + 220), SS * (cy + 330)), SS * 40, fill=GUM + (int(255 * pr),))
            d.polygon(tooth_pts(E.RC_TOOTH, cx, cy - lift, 330, 560, SS), fill=IVORY + (int(255 * pr),), outline=GOLD + (int(255 * pr),), width=5)
        d.line((SS * 540, SS * 560, SS * 540, SS * 1300), fill=GOLD + (120,), width=SS * 2)
        ov = ov.resize((W, H), Image.LANCZOS)
        img = E_composite(img, ov)
        if t >= tl:
            E.put(img, label("LIJEČENJE", 64, GOLD_HI), 290, 1320, 1, E.clamp((t - tl) / 0.2))
        if t >= tr:
            E.put(img, label("VAĐENJE", 64, WHITE), 790, 1320, 1, E.clamp((t - tr) / 0.2))
        if t >= tr + 0.2:
            a = E.clamp((t - tr - 0.2) / 0.2)
            vs = label("ili", 54, GOLD)
            E.put(img, vs, 540, 930, 1, a)
        return push(img, t, self.t0, self.t1, 0.05)


class Extract:
    def __init__(self, t0, t1, marks, title=""):
        self.t0, self.t1, self.m = t0, t1, marks

    def draw(self, t):
        img = background()
        to = self.m.get("out", self.t0 + 0.4)
        lift = E.ease_out(t, to, 0.9)
        ov = Image.new("RGBA", (W * SS, H * SS), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)
        cx, cy = 540, 980
        d.rounded_rectangle((SS * 150, SS * (cy + 40), SS * 930, SS * (cy + 420)), SS * 80, fill=GUM + (255,))
        for nx in (250, 830):
            d.polygon(tooth_pts(E.RC_TOOTH, nx, cy + 60, 230, 470, SS), fill=IVORY + (255,), outline=(205, 196, 180, 255), width=4)
        if lift > 0.05:                                        # dashed outline of the gap left behind
            pts = tooth_pts(E.RC_TOOTH, cx, cy + 60, 300, 560, SS)
            for k in range(0, len(pts), 2):
                a, b = pts[k], pts[(k + 1) % len(pts)]
                d.line((a, b), fill=GOLD_HI + (int(255 * lift),), width=SS * 4)
        d.polygon(tooth_pts(E.RC_TOOTH, cx, cy + 60 - 520 * lift, 300, 560, SS), fill=IVORY + (255,),
                  outline=GOLD + (255,), width=5)
        ov = ov.resize((W, H), Image.LANCZOS)
        img = E_composite(img, ov)
        if lift > 0.6:
            a = E.clamp((lift - 0.6) / 0.3)
            E.put(img, label("?", 220, GOLD_HI, "Black"), 540, 1160, 0.8 + 0.2 * a, a)
        return push(img, t, self.t0, self.t1, 0.07)


KINDS = {"xray": XRay, "root": Root, "restore": Restore, "pain": Pain, "versus": Versus, "extract": Extract}


class Video:
    """real footage as b-roll: centre-cropped to 9:16 (focus 0..1 moves the crop left/right), slow push-in.
    marks: none. Extra event fields: file, start (seconds into the clip), focus."""

    def __init__(self, t0, t1, marks, file, start=0.0, focus=0.5, title=""):
        self.t0, self.t1, self.file, self.start, self.focus = t0, t1, file, float(start), float(focus)
        self.proc, self.next_i, self.last = None, None, None

    def _open(self, i):
        import subprocess
        if self.proc:
            self.proc.kill()
        ss = self.start + i / 30.0
        vf = (f"fps=30,crop='min(iw,ih*9/16)':ih:'(iw-min(iw,ih*9/16))*{self.focus}':0,"
              f"scale={W}:{H}:flags=lanczos,unsharp=5:5:0.5")
        self.proc = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{ss:.3f}", "-i", str(self.file), "-vf", vf,
                                      "-an", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
        self.next_i = i
        self.last = None

    def draw(self, t):
        i = max(0, int(round((t - self.t0) * 30)))
        if self.proc is None or i != self.next_i:
            if not (self.last is not None and i == self.next_i - 1):
                self._open(i)
        if i == self.next_i:
            buf = self.proc.stdout.read(W * H * 3)
            if len(buf) == W * H * 3:
                self.last = np.frombuffer(buf, np.uint8).reshape(H, W, 3).astype(np.float32)
            self.next_i += 1
        img = self.last if self.last is not None else np.zeros((H, W, 3), np.float32)
        return push(img, t, self.t0, self.t1, 0.05)


KINDS["video"] = Video
