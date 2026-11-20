"""dental-gold element library: captions, cards, type lockups, dental animations (drawn, no stock)."""
import json, math, re, subprocess, sys
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter

W, H, FPS = 1080, 1920, 30
FD = str(__import__("pathlib").Path(__file__).resolve().parent.parent / "fonts") + "/"
WHITE = (255, 255, 255)
AQUA = (214, 178, 94)   # brand gold (name kept for the accent role)
GOLD_HI = (238, 205, 128)
RED = AQUA
MUTED = (176, 170, 158)
CARD_BG = (12, 12, 12, 236)
IVORY_OLD = (226, 208, 166)
IVORY = (250, 248, 242)
GUM = (176, 96, 104)
FACE = (580, 494, 209, 298)
CAP_Y = 1010

_fc = {}


def font(size, wt="Bold", fam="Montserrat.ttf"):
    k = (size, wt, fam)
    if k not in _fc:
        f = ImageFont.truetype(FD + fam, size)
        try:
            f.set_variation_by_name(wt)
        except Exception:
            pass
        _fc[k] = f
    return _fc[k]


# ---------- easing ----------
def clamp(x, a=0.0, b=1.0):
    return max(a, min(b, x))


def spring(t, t0, dur=0.45):
    """0->1 with a small overshoot"""
    if t <= t0:
        return 0.0
    u = clamp((t - t0) / dur)
    return 1 - math.exp(-6 * u) * math.cos(7.5 * u) if u < 1 else 1.0


def ease_out(t, t0, dur):
    u = clamp((t - t0) / dur)
    return 1 - (1 - u) ** 3


# ---------- words / anchors ----------
words = []


def norm(w):
    return re.sub(r"[^\wčćšđž]", "", w.lower())


def at(phrase, after=0.0):
    ph = [norm(x) for x in phrase.split()]
    for i in range(len(words) - len(ph) + 1):
        if words[i]["s"] >= after and [norm(words[i + j]["w"]) for j in range(len(ph))] == ph:
            return words[i]["s"]
    raise SystemExit("anchor not found: " + phrase)


# ---------- drawing primitives (RGBA numpy layers) ----------
def rrect_img(w, h, r, fill, border=None):
    im = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, w - 1, h - 1), r, fill=fill, outline=border, width=2 if border else 0)
    return im


def with_shadow(im, blur=22, alpha=0.45, dy=10):
    pad = blur * 2
    w, h = im.size
    out = Image.new("RGBA", (w + pad * 2, h + pad * 2), (0, 0, 0, 0))
    a = im.split()[3].point(lambda v: int(v * alpha))
    sh = Image.new("RGBA", im.size, (0, 0, 0, 255))
    sh.putalpha(a)
    out.paste(sh, (pad, pad + dy), sh)
    out = out.filter(ImageFilter.GaussianBlur(blur))
    out.alpha_composite(im, (pad, pad))
    return out


def text_w(s, f):
    return f.getlength(s)


def check_icon(d, x, y, s, prog, color=AQUA):
    d.ellipse((x, y, x + s, y + s), fill=color + (255,))
    pts = [(x + s * 0.27, y + s * 0.52), (x + s * 0.44, y + s * 0.68), (x + s * 0.74, y + s * 0.34)]
    if prog > 0:
        L1 = math.dist(pts[0], pts[1]); L2 = math.dist(pts[1], pts[2]); L = (L1 + L2) * prog
        if L <= L1:
            e = pts[0][0] + (pts[1][0] - pts[0][0]) * L / L1, pts[0][1] + (pts[1][1] - pts[0][1]) * L / L1
            d.line([pts[0], e], fill=(14, 16, 20, 255), width=max(3, int(s * 0.12)))
        else:
            r = (L - L1) / L2
            e = pts[1][0] + (pts[2][0] - pts[1][0]) * r, pts[1][1] + (pts[2][1] - pts[1][1]) * r
            d.line([pts[0], pts[1], e], fill=(14, 16, 20, 255), width=max(3, int(s * 0.12)), joint="curve")


def box_icon(d, x, y, s):
    d.rounded_rectangle((x, y, x + s, y + s), 10, outline=(120, 126, 136, 255), width=3)


def calendar_icon(d, x, y, s, color=AQUA):
    d.rounded_rectangle((x, y + s * 0.12, x + s, y + s), 10, outline=color + (255,), width=4)
    d.line((x, y + s * 0.38, x + s, y + s * 0.38), fill=color + (255,), width=4)
    for k in (0.28, 0.72):
        d.line((x + s * k, y, x + s * k, y + s * 0.24), fill=color + (255,), width=5)
    for r in range(2):
        for c_ in range(3):
            cx, cy = x + s * (0.22 + 0.28 * c_), y + s * (0.58 + 0.22 * r)
            d.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=color + (255,))


def tooth_icon(d, x, y, s, color=AQUA):
    pts = [(0.15, 0.25), (0.3, 0.08), (0.5, 0.16), (0.7, 0.08), (0.85, 0.25), (0.82, 0.55), (0.72, 0.95),
           (0.6, 0.95), (0.52, 0.62), (0.48, 0.62), (0.4, 0.95), (0.28, 0.95), (0.18, 0.55)]
    d.polygon([(x + s * a, y + s * b) for a, b in pts], outline=color + (255,), width=4)


# ---------- composite helper ----------
def put(img, im, cx, cy, scale=1.0, alpha=1.0):
    if alpha <= 0.01 or scale <= 0.02:
        return
    a = np.asarray(im, dtype=np.float32)
    if abs(scale - 1) > 0.003:
        a = cv2.resize(a, (max(1, int(a.shape[1] * scale)), max(1, int(a.shape[0] * scale))),
                       interpolation=cv2.INTER_LINEAR)
    h, w = a.shape[:2]
    x0, y0 = int(round(cx - w / 2)), int(round(cy - h / 2))
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + w), min(H, y0 + h)
    if xb <= xa or yb <= ya:
        return
    p = a[ya - y0:yb - y0, xa - x0:xb - x0]
    al = p[..., 3:4] / 255.0 * alpha
    roi = img[ya:yb, xa:xb]
    roi[:] = roi * (1 - al) + p[..., :3] * al


def pop(t, t0, t1, dur=0.42):
    """scale/alpha/dy for an element living in [t0, t1]"""
    if t < t0 - 0.02 or t > t1 + 0.25:
        return None
    s = spring(t, t0, dur)
    ex = ease_out(t, t1, 0.22)
    scale = (0.82 + 0.18 * s) * (1 - 0.06 * ex)
    alpha = clamp((t - t0) / 0.12) * (1 - ex)
    dy = (1 - s) * 40
    return scale, alpha, dy


# ---------- elements ----------
class Card:
    """simple card: optional icon + title + sub"""

    def __init__(self, t0, t1, title, sub="", icon=None, cy=1290, w=None):
        self.t0, self.t1, self.cy = t0, t1, cy
        ft, fs = font(54, "Bold"), font(36, "Medium")
        iw = 86 if icon else 0
        tw = max(text_w(title, ft), text_w(sub, fs) if sub else 0)
        cw = int(w or tw + iw + 96)
        ch = 150 if sub else 118
        im = rrect_img(cw, ch, 34, CARD_BG, (255, 255, 255, 34))
        d = ImageDraw.Draw(im)
        x = 46
        if icon:
            s = 62
            iy = (ch - s) // 2
            if icon == "check":
                check_icon(d, x, iy, s, 1.0)
            elif icon == "calendar":
                calendar_icon(d, x, iy, s)
            elif icon == "tooth":
                tooth_icon(d, x, iy, s)
            x += iw
        ty = 26 if sub else (ch - 62) // 2
        d.text((x, ty), title, font=ft, fill=WHITE)
        if sub:
            d.text((x, ty + 68), sub, font=fs, fill=MUTED)
        self.im = with_shadow(im)

    def draw(self, img, t):
        p = pop(t, self.t0, self.t1)
        if p:
            put(img, self.im, 540, self.cy + p[2], p[0], p[1])


class Bar:
    """'Brušenje' meter that fills only a little: minimal"""

    def __init__(self, t0, t1, cy=1300):
        self.t0, self.t1, self.cy = t0, t1, cy

    def draw(self, img, t):
        p = pop(t, self.t0, self.t1)
        if not p:
            return
        cw, ch = 760, 190
        im = rrect_img(cw, ch, 34, CARD_BG, (255, 255, 255, 34))
        d = ImageDraw.Draw(im)
        d.text((46, 28), "Brušenje zuba", font=font(48, "Bold"), fill=WHITE)
        fill = 0.18 * ease_out(t, self.t0 + 0.35, 0.9)
        d.text((cw - 46 - text_w("minimalno", font(38, "SemiBold")), 34), "minimalno",
               font=font(38, "SemiBold"), fill=AQUA)
        x0, y0, x1 = 46, 118, cw - 46
        d.rounded_rectangle((x0, y0, x1, y0 + 26), 13, fill=(255, 255, 255, 40))
        d.rounded_rectangle((x0, y0, x0 + max(26, (x1 - x0) * fill), y0 + 26), 13, fill=AQUA + (255,))
        put(img, with_shadow(im), 540, self.cy + p[2], p[0], p[1])


class Compare:
    """two cards side by side, bullets appear on their own anchors"""

    def __init__(self, t0, t1, left, right, cy=1330):
        self.t0, self.t1, self.cy = t0, t1, cy
        self.cols = [left, right]  # (title, t_title, [(bullet, t_b)], t_highlight)

    def draw(self, img, t):
        if t < self.t0 - 0.02 or t > self.t1 + 0.3:
            return
        ex = ease_out(t, self.t1, 0.25)
        for k, (title, tt, bullets, hl) in enumerate(self.cols):
            s = spring(t, tt, 0.5)
            if s <= 0:
                continue
            cw, ch = 470, 330
            act = hl[0] <= t < hl[1]
            im = rrect_img(cw, ch, 32, CARD_BG, (AQUA + (230,)) if act else (255, 255, 255, 34))
            d = ImageDraw.Draw(im)
            d.text((36, 30), title, font=font(50, "Bold"), fill=AQUA if act else WHITE)
            y = 112
            for b, tb in bullets:
                if t >= tb:
                    a = int(255 * clamp((t - tb) / 0.2))
                    d.ellipse((38, y + 14, 52, y + 28), fill=AQUA + (a,))
                    lines = b.split("\n")
                    for ln in lines:
                        d.text((68, y), ln, font=font(34, "Medium"), fill=WHITE + (a,))
                        y += 44
                    y += 14
            cx = 285 + 510 * k
            slide = (1 - s) * (-260 if k == 0 else 260)
            sc = (1.04 if act else 1.0) * (1 - 0.06 * ex)
            put(img, with_shadow(im), cx + slide, self.cy, sc, clamp((t - tt) / 0.15) * (1 - ex))


class Checklist:
    def __init__(self, t0, t1, title, items, cy=1340):
        self.t0, self.t1, self.title, self.items, self.cy = t0, t1, title, items, cy

    def draw(self, img, t):
        p = pop(t, self.t0, self.t1)
        if not p:
            return
        cw, ch = 900, 120 + 112 * len(self.items)
        im = rrect_img(cw, ch, 36, CARD_BG, (255, 255, 255, 34))
        d = ImageDraw.Draw(im)
        d.text((44, 34), self.title, font=font(40, "Bold"), fill=AQUA)
        y = 116
        for txt, ti in self.items:
            if t >= ti:
                a = clamp((t - ti) / 0.2)
                dx = int((1 - ease_out(t, ti, 0.3)) * 40)
                ck = clamp((t - ti - 0.45) / 0.3)
                if ck > 0:
                    check_icon(d, 44 + dx, y, 58, ck)
                else:
                    box_icon(d, 44 + dx, y, 58)
                d.text((128 + dx, y + 6), txt, font=font(42, "SemiBold"), fill=WHITE + (int(255 * a),))
            y += 112
        put(img, with_shadow(im), 540, self.cy + p[2], p[0], p[1])


class Lockup:
    """big word with small words around (ref 3 style)"""

    def __init__(self, t0, t1, big, above="", below="", cy=CAP_Y + 20, color=None):
        color = color or GOLD_HI
        self.t0, self.t1, self.cy = t0, t1, cy
        fb = font(150, "Black")
        while text_w(big, fb) > 960:
            fb = font(int(fb.size * 0.92), "Black")
        fsm = font(46, "SemiBold")
        bw = int(text_w(big, fb)) + 80
        hh = fb.size + 160
        im = Image.new("RGBA", (bw, hh), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.text((40, 70), big, font=fb, fill=color)
        if above:
            d.text((44, 18), above, font=fsm, fill=WHITE)
        if below:
            d.text((bw - 40 - text_w(below, fsm), 70 + fb.size * 0.98), below, font=fsm, fill=WHITE)
        glow = im.filter(ImageFilter.GaussianBlur(18))
        sh = Image.new("RGBA", im.size, (0, 0, 0, 0))
        a = im.split()[3].filter(ImageFilter.GaussianBlur(10)).point(lambda v: int(v * 0.8))
        sh.putalpha(a)
        out = Image.alpha_composite(sh, glow)
        self.im = Image.alpha_composite(out, im)

    def active(self, t):
        return self.t0 - 0.02 <= t <= self.t1 + 0.15

    def draw(self, img, t):
        if not self.active(t):
            return
        s = spring(t, self.t0, 0.38)
        ex = ease_out(t, self.t1, 0.15)
        sc = (1.25 - 0.25 * s) * (1 + 0.08 * ex)
        put(img, self.im, 540, self.cy, sc, clamp((t - self.t0) / 0.1) * (1 - ex))


class Badge:
    """name tag with a curved arrow pointing at the face"""

    def __init__(self, t0, t1, text):
        self.t0, self.t1 = t0, t1
        f = font(68, "ExtraBold")
        w = int(text_w(text, f)) + 40
        im = Image.new("RGBA", (w, 110), (0, 0, 0, 0))
        d = ImageDraw.Draw(im)
        d.text((20, 10), text, font=f, fill=AQUA)
        a = im.split()[3].filter(ImageFilter.GaussianBlur(8)).point(lambda v: int(v * 0.7))
        sh = Image.new("RGBA", im.size, (0, 0, 0, 255))
        sh.putalpha(a)
        self.im = Image.alpha_composite(sh, im)
        fx, fy, fw, fh = FACE
        self.x, self.y = max(150, fx - 350), max(90, fy - 245)
        self.p2 = (fx - fw / 2 - 70, fy - 25)

    def draw(self, img, t):
        p = pop(t, self.t0, self.t1)
        if not p:
            return
        put(img, self.im, self.x, self.y + p[2], p[0], p[1])
        # arrow: draw progressively
        prog = ease_out(t, self.t0 + 0.15, 0.45) * p[1]
        if prog <= 0:
            return
        L = Image.new("RGBA", (W, 700), (0, 0, 0, 0))
        d = ImageDraw.Draw(L)
        p0 = np.array([self.x + 30, self.y + 60]); p1 = np.array([self.x + 60, self.y + 190]); p2 = np.array(self.p2)
        pts = [tuple((1 - u) ** 2 * p0 + 2 * (1 - u) * u * p1 + u * u * p2) for u in np.linspace(0, prog, 24)]
        d.line(pts, fill=WHITE + (255,), width=7, joint="curve")
        if prog > 0.95:
            tip = np.array(pts[-1]); v = tip - np.array(pts[-3]); v = v / (np.linalg.norm(v) + 1e-6)
            n = np.array([-v[1], v[0]])
            d.polygon([tuple(tip + v * 8), tuple(tip - v * 26 + n * 16), tuple(tip - v * 26 - n * 16)],
                      fill=WHITE + (255,))
        put(img, L, W / 2, 350, 1.0, p[1])


class Strip:
    """red topic strip (hook)"""

    def __init__(self, t0, t1, text, cy=1160):
        self.t0, self.t1, self.cy = t0, t1, cy
        f = font(50, "ExtraBold")
        w = int(text_w(text, f)) + 70
        im = rrect_img(w, 92, 14, RED + (255,))
        ImageDraw.Draw(im).text((35, 16), text, font=f, fill=(12, 12, 12))
        self.im = with_shadow(im, 16, 0.4, 8)

    def draw(self, img, t):
        p = pop(t, self.t0, self.t1)
        if p:
            put(img, self.im, 540, self.cy + p[2], p[0], p[1])


class CommentBar:
    def __init__(self, t0, t1, text, cy=1330):
        self.t0, self.t1, self.text, self.cy = t0, t1, text, cy

    def draw(self, img, t):
        p = pop(t, self.t0, self.t1)
        if not p:
            return
        cw, ch = 900, 128
        im = rrect_img(cw, ch, 64, (250, 250, 252, 245))
        d = ImageDraw.Draw(im)
        d.ellipse((22, 22, 106, 106), fill=AQUA + (255,))
        tooth_icon(d, 40, 38, 48, color=(20, 21, 26))
        n = int(len(self.text) * clamp((t - self.t0 - 0.5) / 1.6))
        s = self.text[:n]
        d.text((128, 40), s or "Dodaj komentar…", font=font(42, "SemiBold"),
               fill=(20, 21, 26) if s else (150, 150, 158))
        if n < len(self.text) and int(t * 2.4) % 2 == 0 and s:
            x = 128 + text_w(s, font(42, "SemiBold")) + 4
            d.line((x, 38, x, 92), fill=(20, 21, 26, 255), width=4)
        done = n >= len(self.text)
        pulse = 1 + 0.12 * math.sin((t - self.t0) * 7) if done else 1
        r = 40 * pulse
        cx, cy = cw - 70, ch / 2
        d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=AQUA + (255,) if done else (200, 204, 212, 255))
        d.polygon([(cx, cy - 18), (cx - 16, cy + 2), (cx + 16, cy + 2)], fill=(12, 12, 12))
        d.line((cx, cy - 4, cx, cy + 20), fill=(12, 12, 12), width=7)
        put(img, with_shadow(im), 540, self.cy + p[2], p[0], p[1])


# ---------- dental animations ----------
SS = 2  # supersampling for smooth edges


def smooth_poly(pts, n=4):
    """Chaikin smoothing of a closed polygon"""
    p = [tuple(map(float, x)) for x in pts]
    for _ in range(n):
        q = []
        for i in range(len(p)):
            a, b = p[i], p[(i + 1) % len(p)]
            q += [(0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]),
                  (0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1])]
        p = q
    return p


INCISOR = smooth_poly([(-0.36, -0.5), (0.36, -0.5), (0.47, -0.1), (0.5, 0.42), (0.44, 0.5),
                       (-0.44, 0.5), (-0.5, 0.42), (-0.47, -0.1)], 6)


def shape_at(poly, cx, cy, w, h):
    return [(SS * (cx + x * w), SS * (cy + y * h)) for x, y in poly]


def mix(a, b, u):
    return tuple(int(a[i] + (b[i] - a[i]) * u) for i in range(3))


def card_with_title(cw, ch, title):
    im = rrect_img(cw * SS, ch * SS, 34 * SS, CARD_BG, (AQUA + (90,)))
    d = ImageDraw.Draw(im)
    d.text((44 * SS, 28 * SS), title, font=font(46 * SS, "Bold"), fill=WHITE)
    return im, d


def blend_poly(im, pts, fill, outline=None, width=0):
    L = Image.new("RGBA", im.size, (0, 0, 0, 0))
    ImageDraw.Draw(L).polygon(pts, fill=fill, outline=outline, width=width)
    im.alpha_composite(L)


def sparkle(d, x, y, r, a):
    if a <= 0:
        return
    col = GOLD_HI + (int(255 * a),)
    d.polygon([(x, y - r), (x + r * 0.22, y - r * 0.22), (x + r, y), (x + r * 0.22, y + r * 0.22),
               (x, y + r), (x - r * 0.22, y + r * 0.22), (x - r, y), (x - r * 0.22, y - r * 0.22)], fill=col)


class VeneerFront:
    """front view: two incisors, a veneer slides down onto each and the smile brightens"""

    def __init__(self, t0, t1, title, cy=1395):
        self.t0, self.t1, self.title, self.cy = t0, t1, title, cy

    def draw(self, img, t):
        p = pop(t, self.t0, self.t1)
        if not p:
            return
        cw, ch = 760, 450
        im, d = card_with_title(cw, ch, self.title)
        # gum arch
        d.rounded_rectangle((SS * 150, SS * 112, SS * 610, SS * 186), SS * 36, fill=GUM + (255,))
        tw, th = 190, 250
        centers = [(283, 295), (477, 295)]
        for k, (x, y) in enumerate(centers):
            tv = self.t0 + 0.45 + 0.35 * k
            u = ease_out(t, tv, 0.6)
            d.polygon(shape_at(INCISOR, x, y, tw, th), fill=mix(IVORY_OLD, IVORY, u) + (255,))
            if t >= tv - 0.01:
                off = (1 - u) * -170
                a = clamp((t - tv) / 0.15)
                blend_poly(im, shape_at(INCISOR, x, y + off, tw * 1.05, th * 1.03),
                           (255, 255, 255, int(110 * a)), AQUA + (int(255 * a),), 3 * SS)
        # shine sweep after both veneers land
        ts = self.t0 + 1.5
        if t > ts:
            g = clamp((t - ts) / 0.6)
            xs = 180 + g * 420
            band = Image.new("RGBA", im.size, (0, 0, 0, 0))
            ImageDraw.Draw(band).polygon([(SS * (xs - 10), SS * 160), (SS * (xs + 40), SS * 160),
                                          (SS * (xs - 20), SS * 430), (SS * (xs - 70), SS * 430)],
                                         fill=(255, 255, 255, int(150 * (1 - abs(2 * g - 1)))))
            mask = Image.new("L", im.size, 0)
            md = ImageDraw.Draw(mask)
            for x, y in centers:
                md.polygon(shape_at(INCISOR, x, y, tw, th), fill=255)
            band.putalpha(Image.fromarray(np.minimum(np.array(mask), np.array(band.split()[3]))))
            im.alpha_composite(band)
            sparkle(d, SS * 590, SS * 200, SS * 22 * math.sin(math.pi * g), 1.0)
            sparkle(d, SS * 180, SS * 400, SS * 14 * math.sin(math.pi * clamp(g * 1.3 - 0.3)), 1.0)
        im = im.resize((cw, ch), Image.LANCZOS)
        put(img, with_shadow(im), 540, self.cy + p[2], p[0], p[1])


# side profile of an upper incisor (x right = toward the tongue), labial surface on the left
LAB = [(150, 30), (136, 110), (128, 170), (124, 230), (126, 290), (134, 340), (146, 372)]
LING = [(162, 376), (178, 340), (205, 285), (226, 240), (214, 200), (190, 150), (178, 90), (170, 30)]


def labial_offset(dx):
    """labial points shifted inward below the gum line (y > 170)"""
    return [(x + (dx if y >= 170 else dx * clamp((y - 120) / 50)), y) for x, y in LAB]


class VeneerSide:
    """profile: a thin front layer is marked in gold, removed, then the veneer takes its place"""

    def __init__(self, t0, t1, title, cy=1395):
        self.t0, self.t1, self.title, self.cy = t0, t1, title, cy

    def draw(self, img, t):
        p = pop(t, self.t0, self.t1)
        if not p:
            return
        cw, ch = 760, 450
        im, d = card_with_title(cw, ch, self.title)
        ox, oy = 110, 60
        tr = lambda pts: [(SS * (ox + 175 + (x - 175) * 1.45), SS * (y + oy)) for x, y in pts]
        t_mark, t_cut, t_ven = self.t0 + 0.35, self.t0 + 0.95, self.t0 + 1.45
        cut = ease_out(t, t_cut, 0.35)
        lab_in = labial_offset(13)
        lab_now = [(a[0] + (b[0] - a[0]) * cut, a[1]) for a, b in zip(LAB, lab_in)]
        # gum
        d.polygon(tr([(80, 120), (128, 150), (128, 205), (80, 230)]), fill=GUM + (255,))
        d.polygon(tr([(270, 120), (214, 150), (220, 210), (270, 235)]), fill=GUM + (255,))
        d.polygon(tr(smooth_poly(lab_now + LING, 4)), fill=IVORY + (255,))
        # gold layer that will be removed
        layer = [p_ for p_ in LAB if p_[1] >= 150] + [p_ for p_ in lab_in if p_[1] >= 150][::-1]
        a_mark = clamp((t - t_mark) / 0.2) * (1 - cut)
        if a_mark > 0:
            blend_poly(im, tr(layer), AQUA + (int(240 * a_mark),))
        # veneer slides in from the left
        if t >= t_ven:
            u = ease_out(t, t_ven, 0.55)
            dx = (1 - u) * -150
            ven = [(x + dx, y) for x, y in layer]
            blend_poly(im, tr(ven), (255, 255, 255, 240), AQUA + (255,), 2 * SS)
            sparkle(d, SS * (ox + 100), SS * (oy + 250), SS * 16 * math.sin(math.pi * clamp((t - t_ven - 0.5) / 0.6)), 1)
        # labels
        fl = font(32 * SS, "SemiBold")
        if a_mark > 0:
            d.text((SS * 470, SS * 250), "tanak sloj", font=fl, fill=AQUA + (int(255 * a_mark),))
            d.line((SS * 462, SS * 270, SS * 300, SS * 300), fill=AQUA + (int(255 * a_mark),), width=3 * SS)
        if t >= t_ven + 0.4:
            a = clamp((t - t_ven - 0.4) / 0.25)
            d.text((SS * 470, SS * 250), "vinir", font=font(40 * SS, "Bold"), fill=WHITE + (int(255 * a),))
        im = im.resize((cw, ch), Image.LANCZOS)
        put(img, with_shadow(im), 540, self.cy + p[2], p[0], p[1])




# front cross-section of a single-rooted tooth (x right, y down, centre at 0,0; about 1 x 1 units)
RC_TOOTH = smooth_poly([(-0.46, -0.30), (-0.36, -0.47), (-0.14, -0.40), (0, -0.44), (0.14, -0.40), (0.36, -0.47),
                        (0.46, -0.30), (0.44, -0.12), (0.34, 0.0), (0.24, 0.25), (0.12, 0.47), (0.0, 0.5),
                        (-0.12, 0.47), (-0.24, 0.25), (-0.34, 0.0), (-0.44, -0.12)], 4)
RC_PULP = smooth_poly([(-0.20, -0.25), (-0.12, -0.31), (0, -0.28), (0.12, -0.31), (0.20, -0.25), (0.16, -0.11),
                       (0.06, 0.04), (0.035, 0.40), (0, 0.44), (-0.035, 0.40), (-0.06, 0.04), (-0.16, -0.11)], 3)
PULP = (150, 64, 74)      # inflamed nerve, muted rose
BONE = (86, 76, 66)
EMPTY = (36, 28, 28)


class RootCanal:
    """cross-section: the inflamed nerve is removed top-down, a gold file cleans the canal, the canal is
    filled with gold from the apex up, a filling closes the crown and the tooth stays."""

    def __init__(self, t0, t1, title, remove=None, clean=None, fill=None, keep=None, cy=1385):
        self.t0, self.t1, self.title, self.cy = t0, t1, title, cy
        self.tr = remove if remove is not None else t0 + 0.6
        self.tc = clean if clean is not None else self.tr + 1.6
        self.tf = fill if fill is not None else self.tc + 1.6
        self.tk = keep if keep is not None else self.tf + 1.4

    def clip_rows(self, layer, y_from, y_to):
        a = np.array(layer)
        a[:max(0, int(y_from)), :, 3] = 0
        a[int(y_to):, :, 3] = 0
        return Image.fromarray(a)

    def draw(self, img, t):
        p = pop(t, self.t0, self.t1)
        if not p:
            return
        cw, ch = 760, 540
        im, d = card_with_title(cw, ch, self.title)
        cx, cy, tw, th = 235, 328, 230, 390
        y_top, y_bot = cy - 0.31 * th, cy + 0.44 * th            # pulp extent
        # bone and gum around the root
        d.rounded_rectangle((SS * 70, SS * (cy + 0.02 * th), SS * 400, SS * (ch - 18)), SS * 30, fill=BONE + (255,))
        d.polygon([(SS * x, SS * y) for x, y in [(70, cy - 0.02 * th), (cx - 0.40 * tw, cy - 0.06 * th),
                   (cx - 0.33 * tw, cy + 0.10 * th), (70, cy + 0.12 * th)]], fill=GUM + (255,))
        d.polygon([(SS * x, SS * y) for x, y in [(400, cy - 0.02 * th), (cx + 0.40 * tw, cy - 0.06 * th),
                   (cx + 0.33 * tw, cy + 0.10 * th), (400, cy + 0.12 * th)]], fill=GUM + (255,))
        d.polygon(shape_at(RC_TOOTH, cx, cy, tw, th), fill=IVORY + (255,), outline=(205, 196, 180, 255), width=SS * 2)
        pulp = shape_at(RC_PULP, cx, cy, tw, th)
        u_rm = ease_out(t, self.tr, 0.9)                         # nerve removal, top-down
        y_cut = y_top + (y_bot - y_top + 4) * u_rm
        # empty canal after removal, then cleaned (lighter, gold outline)
        if u_rm > 0:
            cl = ease_out(t, self.tc, 1.2)
            L = Image.new("RGBA", im.size, (0, 0, 0, 0))
            ImageDraw.Draw(L).polygon(pulp, fill=mix(EMPTY, (70, 60, 52), cl) + (255,),
                                      outline=AQUA + (int(200 * cl),), width=SS * 2)
            im.alpha_composite(self.clip_rows(L, 0, SS * y_cut))
        # inflamed nerve with a pain pulse
        L = Image.new("RGBA", im.size, (0, 0, 0, 0))
        pulse = 0.5 + 0.5 * math.sin((t - self.t0) * 7.0) if t < self.tr else 0
        ImageDraw.Draw(L).polygon(pulp, fill=mix(PULP, (196, 92, 100), pulse) + (255,))
        im.alpha_composite(self.clip_rows(L, SS * y_cut, im.size[1]))
        # access opening in the crown
        if t >= self.tr - 0.2:
            a = clamp((t - self.tr + 0.2) / 0.25)
            d.ellipse((SS * (cx - 22), SS * (cy - 0.47 * th), SS * (cx + 22), SS * (cy - 0.30 * th)),
                      fill=EMPTY + (int(255 * a),))
        # gold file moving in and out of the canal
        if self.tc <= t < self.tf:
            k = (t - self.tc) / max(0.1, self.tf - self.tc)
            depth = 0.5 - 0.5 * math.cos(k * 3 * 2 * math.pi)
            tip = y_top + 20 + (y_bot - y_top - 30) * depth
            top = cy - 0.53 * th
            d.line((SS * cx, SS * top, SS * cx, SS * tip), fill=AQUA + (255,), width=SS * 4)
            for yy in np.arange(top + 30, tip - 6, 14):
                d.line((SS * (cx - 5), SS * yy, SS * (cx + 5), SS * (yy + 5)), fill=GOLD_HI + (255,), width=SS * 2)
            d.rounded_rectangle((SS * (cx - 13), SS * (top - 22), SS * (cx + 13), SS * top), SS * 6,
                                fill=GOLD_HI + (255,))
        # gold filling from the apex up, then the crown is closed
        if t >= self.tf:
            u = ease_out(t, self.tf, 1.0)
            y_fill = y_bot + 4 - (y_bot - y_top + 4) * u
            L = Image.new("RGBA", im.size, (0, 0, 0, 0))
            ImageDraw.Draw(L).polygon(pulp, fill=AQUA + (255,))
            im.alpha_composite(self.clip_rows(L, SS * y_fill, im.size[1]))
            cap = clamp((t - self.tf - 0.9) / 0.35)
            if cap > 0:
                d.ellipse((SS * (cx - 24), SS * (cy - 0.48 * th), SS * (cx + 24), SS * (cy - 0.29 * th)),
                          fill=mix(IVORY_OLD, IVORY, cap) + (int(255 * cap),), outline=AQUA + (int(255 * cap),),
                          width=SS * 2)
        if t >= self.tk:
            g = clamp((t - self.tk) / 0.7)
            sparkle(d, SS * (cx + 0.42 * tw), SS * (cy - 0.50 * th), SS * 20 * math.sin(math.pi * g), 1)
            sparkle(d, SS * (cx - 0.46 * tw), SS * (cy - 0.20 * th), SS * 13 * math.sin(math.pi * clamp(g * 1.3 - 0.3)), 1)
        # step label on the right
        steps = [(self.t0, "Oboljeli živac"), (self.tr, "Uklanja se živac"), (self.tc, "Kanal se čisti"),
                 (self.tf, "Kanal se zatvara"), (self.tk, "Zub ostaje")]
        cur = max(k for k, (ts, _) in enumerate(steps) if t >= ts - 1e-6 or k == 0)
        ts, label = steps[cur]
        a = clamp((t - ts) / 0.25) if cur else 1.0
        fx, fy = 430, 270
        d.text((SS * fx, SS * (fy - 52)), f"{cur + 1}/5", font=font(28 * SS, "SemiBold"),
               fill=AQUA + (int(255 * a),))
        words_ = label.split(" ", 1)
        d.text((SS * fx, SS * fy + SS * 10 * (1 - a)), words_[0], font=font(44 * SS, "Bold"), fill=WHITE + (int(255 * a),))
        if len(words_) > 1:
            d.text((SS * fx, SS * (fy + 54) + SS * 10 * (1 - a)), words_[1], font=font(44 * SS, "Bold"),
                   fill=WHITE + (int(255 * a),))
        im = im.resize((cw, ch), Image.LANCZOS)
        put(img, with_shadow(im), 540, self.cy + p[2], p[0], p[1])
