"""/ui-depth-reel: liquid-glass app windows float in true 3D depth around the speaker. They rush in from deep space
with motion blur, the back ones sit behind him and the front ones in front, and the camera orbits so every window
turns and parallaxes, with the far ones thrown out of focus. A prompt window types each idea letter by letter, a
cursor glides in and presses Generate, and for the payoff a glass window holding his own freeze-frame flies
toward camera while focus racks onto it."""
import math
import subprocess

import cv2
import numpy as np

import core as c
import fx
from plan import Plan, norm

PIV = np.float32([540, 960])
PERSP = 2000.0
ACC = (92, 150, 255)       # electric blue
ACC2 = (168, 108, 255)     # violet end of the button gradient
INK = (255, 255, 255)
DIM = (178, 186, 204)


# ---------- small helpers ----------

def track(keys, t, names):
    """Keyframed values: each key eases from the current value to its targets (like fx.Cam.at)."""
    cur = {n: keys[0].get(n, 0.0) for n in names}
    for k in keys[1:]:
        if t < k["t"]:
            break
        mode, d = k.get("ease", "expo"), k.get("dur", 0.5)
        if mode == "spring":
            e = fx.spring(t, k["t"], k.get("freq", 1.6), k.get("damp", 0.55))
        elif mode == "smooth":
            e = fx.smooth(t, k["t"], d)
        else:
            e = fx.expo(t, k["t"], d)
        cur = {n: cur[n] + (k.get(n, cur[n]) - cur[n]) * e for n in names}
    return cur


def grab_frames(clip, idxs, w, h):
    """Decode a few frames in one ffmpeg pass (downscaled) -> {index: uint8 RGB}."""
    idxs = sorted(set(int(min(max(i, 0), clip.n - 1)) for i in idxs))
    sel = "+".join(f"eq(n\\,{i})" for i in idxs)
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", clip.src, "-vf", f"select={sel},scale={w}:{h}",
                          "-vsync", "0", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True).stdout
    n = len(raw) // (w * h * 3)
    out = {}
    for k, i in enumerate(idxs[:n]):
        out[i] = np.frombuffer(raw[k * w * h * 3:(k + 1) * w * h * 3], np.uint8).reshape(h, w, 3)
    return out


def face_crop(img, face, k, w, h, span=2.3, at=0.42):
    """Crop around a face (face in source px, img downscaled by k) to w x h, float32."""
    fx_, fy, fw, fh = face
    ch = fh * span * k
    cw = ch * w / h
    x0 = int(c.clamp(fx_ * k - cw / 2, 0, img.shape[1] - cw))
    y0 = int(c.clamp(fy * k - ch * at, 0, img.shape[0] - ch))
    crop = img[y0:y0 + int(ch), x0:x0 + int(cw)]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_AREA).astype(np.float32)


def rgb_patch(img, r):
    """Premultiplied RGBA patch of an RGB image with rounded corners."""
    m = c.rrect(img.shape[1], img.shape[0], r)[..., None]
    return np.concatenate([img * m, m * 255], -1).astype(np.float32)


def grad_pill(w, h, c0, c1):
    g = fx.gradient(h, w, c0, c1, angle=0)
    return rgb_patch(g, h // 2)


def sparkle(size, color=(255, 255, 255)):
    """Four-point AI sparkle glyph."""
    s = size
    ys, xs = fx.grid(s, s)
    u, v = np.abs(xs - s / 2 + 0.5) / (s / 2), np.abs(ys - s / 2 + 0.5) / (s / 2)
    m = np.clip((1 - (np.sqrt(u) + np.sqrt(v))) * 6, 0, 1)
    return c.fill(m, color)


def cursor_patch(scale=1.7):
    pts = np.float32([(0, 0), (0, 34), (8.5, 26.5), (14.5, 40), (20.5, 37.5), (14.5, 24.5), (26, 24.5)]) * scale
    pad = 8
    w, h = int(pts[:, 0].max()) + 2 * pad, int(pts[:, 1].max()) + 2 * pad
    sh = 4
    p = ((pts + pad) * (1 << sh)).astype(np.int32)
    ink = np.zeros((h, w), np.uint8)
    cv2.polylines(ink, [p], True, 255, int(3 * scale), cv2.LINE_AA, shift=sh)
    body = np.zeros((h, w), np.uint8)
    cv2.fillPoly(body, [p], 255, cv2.LINE_AA, shift=sh)
    out = c.fill(np.maximum(ink, body).astype(np.float32) / 255, (20, 20, 24))
    c.over(out, c.fill(cv2.erode(body, np.ones((3, 3), np.uint8), iterations=int(scale + 1)).astype(np.float32) / 255,
                       (255, 255, 255)))
    return out, pad


# ---------- the style ----------

class Style:
    def __init__(self, clip, opts):
        self.clip = clip
        self.P = P = Plan(clip, opts)
        self.fr = c.Framing(clip, 1215)
        B = P.beats
        self.B = B
        nb = len(B)
        self.T_end = min(clip.n, len(clip.face)) / clip.fps
        self.pay_t = B[-1]["key_t"] if nb > 1 else self.T_end * 0.75
        pb = B[-2] if nb > 1 else B[0]
        self.press_t = min(pb["key_t"] + 0.5, self.pay_t - 0.14)
        self.press_t = max(self.press_t, pb["key_t"] + 0.2)
        # typing schedule: one label per beat (the last one lands on the payoff window instead)
        self.types = []
        for k, b in enumerate(B[:-1] if nb > 1 else B):
            self.types.append({"t": max(b["key_t"] - 0.1, 0.3), "s": b["label"], "sub": b.get("sub", ""), "k": k})
        # camera orbit: opening swing, a swing on every middle beat (alternating), push on the press, settle on payoff
        ok = [{"t": 0, "yaw": -26, "pitch": 9, "dolly": -200},
              {"t": 0.0, "yaw": -6, "pitch": 2.5, "dolly": 0, "ease": "expo", "dur": 0.9}]
        side = 1
        for b in B[1:-2] if nb > 3 else B[1:2] if nb == 3 else []:
            ok.append({"t": b["key_t"] - 0.05, "yaw": 11 * side, "pitch": -3 * side, "ease": "spring", "freq": 1.1,
                       "damp": 0.62})
            side = -side
        ok.append({"t": pb["key_t"], "dolly": 170, "ease": "smooth", "dur": 0.9})
        ok.append({"t": self.pay_t - 0.05, "yaw": 0, "pitch": 0, "dolly": 60, "ease": "smooth", "dur": 0.8})
        self.orbit = ok
        ck = [{"t": 0, "zoom": 1.13, "dx": -60}, {"t": 0.0, "zoom": 1.0, "dx": -18, "dur": 0.95}]
        for k in ok[2:]:
            q = {"t": k["t"], "ease": k.get("ease", "expo"), "dur": k.get("dur", 0.5), "freq": k.get("freq", 1.6),
                 "damp": k.get("damp", 0.55)}
            if "yaw" in k:
                q["dx"] = k["yaw"] * 3
            if "dolly" in k:
                q["zoom"] = 1.0 + k["dolly"] / 2600
            ok2 = dict(q)
            ck.append(ok2)
        self.cam = fx.Cam(self.fr, ck, drift=3)
        # footage stills: thumbnails for the timeline + the payoff freeze-frame
        k4 = 0.25
        tw, th = int(clip.sw * k4) // 2 * 2, int(clip.sh * k4) // 2 * 2
        self.k4 = tw / clip.sw
        thumb_t = [b["key_t"] + 0.25 for b in B][:4]
        while len(thumb_t) < 3:
            thumb_t.append(thumb_t[-1] + 0.5)
        self.freeze_i = int(min(clip.n - 1, max(0, (B[-2]["key_t"] if nb > 1 else B[0]["key_t"]) * clip.fps + 4)))
        idx = [int(min(clip.n - 1, tt * clip.fps)) for tt in thumb_t] + [self.freeze_i]
        fr_ = grab_frames(clip, idx, tw, th)
        self.thumbs = []
        for i in idx[:-1]:
            img = fr_.get(i)
            if img is not None:
                self.thumbs.append(rgb_patch(face_crop(img, clip.face[i], self.k4, 132, 168), 14))
        fz = fr_.get(self.freeze_i)
        self.freeze = face_crop(fz, clip.face[self.freeze_i], self.k4, 552, 600, span=2.6, at=0.40) \
            if fz is not None else np.full((600, 552, 3), 40, np.float32)
        self.freeze = c.grade(self.freeze, contrast=1.1, sat=1.08, warm=0.015)
        self.cursor, self.cpad = cursor_patch()
        self.hotw = [w for w in clip.words if norm(w["w"]) in P.hot][:8]
        self.env = np.nan_to_num(np.asarray(clip.env, np.float32))  # core leaves NaN past the audio end
        self._tx = {}
        self.wins = self._layout()

    # ---- type cache ----
    def tx(self, s, key="grotesk", size=40, color=INK, wght=None, tracking=0.0):
        kk = (s, key, size, color, wght, tracking)
        if kk not in self._tx:
            self._tx[kk] = fx.text(s, key, size, color, wght, tracking)
        return self._tx[kk]

    # ---- windows ----
    def _layout(self):
        def W(name, w, h, sx, sy, z, rx, ry, rz, order, layer, r=34):
            s = PERSP / (PERSP + z)
            return {"name": name, "w": w, "h": h, "C": np.float32([(sx - PIV[0]) / s, (sy - PIV[1]) / s, z]),
                    "rot": (rx, ry, rz), "order": order, "layer": layer, "r": r,
                    "mask": c.rrect(w, h, r)}
        return [
            W("audio", 860, 330, 600, 300, 1500, 4, -6, 0, 0, "back"),
            W("timeline", 540, 380, 222, 560, 650, 6, 16, -2, 1, "back"),
            W("tokens", 500, 360, 868, 520, 520, 6, -16, 2, 2, "back"),
            W("look", 290, 250, 168, 1060, -250, -4, 18, 1.5, 3, "front", 30),
            W("preview", 280, 380, 905, 1010, -300, -4, -18, -1.5, 4, "front", 30),
            W("prompt", 800, 300, 540, 1452, -230, 12, 0, 0, 5, "front", 38),
        ]

    def chrome(self, w, h, title, tsize=30, right=None):
        p = c.layer(h, w)
        for k, col in enumerate(((255, 95, 87), (254, 188, 46), (40, 200, 64))):
            cv2.circle(p, (30 + 24 * k, 31), 8, (*[v for v in col], 255), -1, cv2.LINE_AA)
        p[:, :, :3] *= p[:, :, 3:4] / 255  # premultiply the dots
        c.over(p, self.tx(title, "mono", tsize, (232, 236, 245), 560, 0.04), 108, 31 - tsize * 0.66)
        if right:
            rp = self.tx(right, "mono", tsize - 2, DIM, 500, 0.06)
            c.over(p, rp, w - 26 - rp.shape[1], 31 - tsize * 0.64)
        c.over(p, c.fill(np.ones((1, w - 2), np.float32), (255, 255, 255), 0.14), 1, 60)
        return p

    def content(self, win, t, i, live):
        n = win["name"]
        w, h = win["w"], win["h"]
        B = self.B
        if n == "prompt":
            k = max([j for j, ty in enumerate(self.types) if ty["t"] <= t] or [0])
            ty = self.types[k]
            p = self.chrome(w, h, "Prompt", 30, f"{ty['k'] + 1:02d}/{len(B):02d}")
            nch = int(max(0.0, t - ty["t"]) * 24)
            s = ty["s"][:nch]
            old = self.types[k - 1]["s"] if k > 0 else ""
            if k > 0 and t < ty["t"]:
                s = old
            sel = k > 0 and ty["t"] - 0.16 <= t < ty["t"] + 0.02
            if sel:
                s = old
            if ty["sub"]:
                c.over(p, self.tx(ty["sub"], "mono", 28, ACC, 600, 0.16), 36, 78)
            if s:
                lab = self.tx(s, "grotesk", 60, INK, 720)
                if sel:
                    c.over(p, c.fill(c.rrect(lab.shape[1] + 12, 70, 8), ACC, 0.55), 30, 122)
                c.over(p, lab, 34, 118)
                cx = 34 + lab.shape[1] - 2
            else:
                cx = 40
            blink = 0.5 + 0.5 * math.cos(t * 2 * math.pi * 1.6)
            if not sel and (t - ty["t"] < len(ty["s"]) / 24 + 0.05 or blink > 0.3):
                c.over(p, c.fill(np.ones((62, 5), np.float32), ACC), cx + 4, 126)
            c.over(p, self.tx("4K  ·  24 FPS", "mono", 28, DIM, 500, 0.08), 36, 232)
            # generate button: squash + ripple on the press
            bw, bh = 236, 74
            bx, by = w - 36 - bw, 210
            pp = c.ease_out_cubic(c.prog(t, self.press_t - 0.06, 0.1))
            sc = 1 - 0.11 * pp
            if t > self.press_t + 0.05:
                sc = 0.89 + 0.11 * fx.spring(t, self.press_t + 0.05, 2.4, 0.33)
            btn = grad_pill(bw, bh, ACC, ACC2)
            hot = math.exp(-((t - self.press_t) / 0.18) ** 2)
            if hot > 0.02:
                btn[..., :3] += btn[..., 3:4] / 255 * 70 * hot
            lab = self.tx("Generate", "grotesk", 34, INK, 640)
            sp = sparkle(26)
            c.over(btn, sp, 30, bh / 2 - 13)
            c.over(btn, lab, 66, bh / 2 - lab.shape[0] / 2 - 2)
            rp = c.prog(t, self.press_t, 0.55)
            if 0 < rp < 1:
                rr = 20 + 190 * c.ease_out_expo(rp)
                ys, xs = fx.grid(bh, bw)
                d = np.hypot(xs - bw * 0.45, ys - bh / 2)
                ring = np.exp(-((d - rr) / 9) ** 2) * (1 - rp) * 0.9 + (d < rr) * 0.22 * (1 - rp)
                btn = btn + c.fill(ring * btn[..., 3] / 255, (255, 255, 255))
                btn[..., 3] = np.minimum(btn[..., 3], 255)
            fx.put(p, btn, bx + bw / 2, by + bh / 2, sc, shadow=0.35, sblur=10, sdy=6)
            self.btn_local = (bx + bw / 2, by + bh / 2)
            return p
        if n == "preview":
            done = c.prog(t, self.press_t + 0.05, 0.5)
            p = self.chrome(w, h, "Preview", 30)
            pw, ph = w - 40, 236
            if live is not None:
                c.over(p, rgb_patch(live, 16), 20, 76)
            # status line
            if t < self.press_t:
                cv2.circle(p, (32, 342), 7, (255, 70, 70, 255), -1, cv2.LINE_AA)
                tc = f"00:{int(t):02d}:{int((t % 1) * 25):02d}"
                c.over(p, self.tx(tc, "mono", 28, INK, 560, 0.02), 50, 324)
            else:
                e = c.ease_out_expo(done)
                c.over(p, c.fill(c.rrect(pw, 10, 5), (255, 255, 255), 0.18), 20, 330)
                if e > 0.02:
                    c.over(p, c.fill(c.rrect(max(10, int(pw * e)), 10, 5), ACC), 20, 330)
                txt = "DONE" if done >= 1 else f"{int(e * 100):d}%"
                c.over(p, self.tx(txt, "mono", 28, INK, 600, 0.1), 20, 284)
            return p
        if n == "look":
            p = self.chrome(w, h, "Look", 30)
            names = ("LENS", "LIGHT", "GRADE")
            for j, nm in enumerate(names):
                y = 92 + j * 54
                c.over(p, self.tx(nm, "mono", 28, DIM, 520, 0.1), 20, y - 4)
                x0, x1 = 128, w - 26
                pos = 0.3 + 0.12 * j
                for k, b in enumerate(self.B):
                    e = fx.spring(t, b["key_t"] + 0.04 * j, 1.8, 0.5)
                    tgt = 0.25 + 0.65 * ((k * 0.37 + j * 0.29 + 0.2) % 1)
                    pos = pos + (tgt - pos) * e
                yy = y + 16
                c.over(p, c.fill(c.rrect(x1 - x0, 6, 3), (255, 255, 255), 0.2), x0, yy - 3)
                kx = x0 + (x1 - x0) * pos
                c.over(p, c.fill(c.rrect(int(kx - x0), 6, 3), ACC), x0, yy - 3)
                cv2.circle(p, (int(kx), int(yy)), 12, (255, 255, 255, 255), -1, cv2.LINE_AA)
            return p
        if n == "timeline":
            p = self.chrome(w, h, "Timeline", 38)
            x = 22
            for th in self.thumbs[:3]:
                c.over(p, th, x, 92)
                x += th.shape[1] + 30
            dur = self.T_end
            px = 22 + (w - 44) * c.clamp(t / dur)
            c.over(p, c.fill(c.rrect(w - 44, 14, 7), (255, 255, 255), 0.16), 22, 300)
            c.over(p, c.fill(c.rrect(max(14, int(px - 22)), 14, 7), ACC), 22, 300)
            c.over(p, c.fill(np.ones((228, 4), np.float32), (255, 255, 255)), px - 2, 82)
            cv2.circle(p, (int(px), 82), 9, (255, 255, 255, 255), -1, cv2.LINE_AA)
            return p
        if n == "tokens":
            p = self.chrome(w, h, "Tokens", 38)
            x, y = 22, 86
            for wd in self.hotw:
                s = norm(wd["w"]).upper() if len(norm(wd["w"])) <= 3 else wd["w"].strip(" ,.").lower()
                lit = fx.smooth(t, wd["s"] - 0.06, 0.18)
                lab = self.tx(s, "grotesk", 40, INK, 620)
                cw, chh = lab.shape[1] + 34, 64
                if x + cw > w - 18:
                    x, y = 22, y + chh + 16
                if y + chh > h - 10:
                    break
                c.over(p, c.fill(c.rrect(cw, chh, chh // 2), (255, 255, 255), 0.12), x, y)
                if lit > 0.01:
                    c.over(p, grad_pill(cw, chh, ACC, ACC2), x, y, lit)
                c.over(p, lab, x + 17, y + chh / 2 - lab.shape[0] / 2 - 1, 0.55 + 0.45 * lit)
                x += cw + 14
            return p
        if n == "audio":
            p = self.chrome(w, h, "Audio", 40)
            nbars = 34
            fps = self.clip.fps
            for j in range(nbars):
                fi = int(c.clamp(t * fps + (j - nbars + 6), 0, self.clip.n - 1))
                v = float(self.env[fi])
                bh_ = int(18 + 170 * v)
                x = 26 + j * ((w - 52) / nbars)
                c.over(p, c.fill(c.rrect(12, bh_, 6), ACC if j > nbars - 8 else (255, 255, 255), 0.85), x,
                       180 - bh_ / 2 + 15)
            return p
        return c.layer(h, w)

    # ---- projection ----
    @staticmethod
    def quad(w, h, C, rot, R, dolly):
        cs = np.float32([[-w / 2, -h / 2, 0], [w / 2, -h / 2, 0], [w / 2, h / 2, 0], [-w / 2, h / 2, 0]])
        P = cs @ fx.rot3(*rot).T + C
        P = P @ R.T
        z = P[:, 2] + PERSP - dolly
        if (z < 250).any():
            return None, None
        return PIV + P[:, :2] * (PERSP / z)[:, None], z

    def pose(self, win, t, cam):
        """Window centre / rotation / opacity at time t (fly-in, payoff recede)."""
        C = win["C"].copy()
        rx, ry, rz = win["rot"]
        tin = -0.15 + win["order"] * 0.045
        e = fx.spring(t, tin, 1.9, 0.62)
        sgn = 1 if C[0] < 0 else -1
        C[2] += 3600 * (1 - e)
        C[1] += 260 * (1 - e) * (1 if C[1] > 0 else -1)
        ry += sgn * 48 * (1 - e)
        rz += sgn * 14 * (1 - e)
        op = c.clamp((t - tin) / 0.07)
        ph = win["order"] * 1.7  # idle float: every window bobs on its own phase
        C[1] += 11 * math.sin(t * 1.25 + ph)
        C[0] += 6 * math.sin(t * 0.9 + ph * 1.3)
        rx += 1.6 * math.sin(t * 1.1 + ph)
        ry += 2.2 * math.sin(t * 0.8 + ph * 0.7)
        q = fx.smooth(t, self.pay_t - 0.08, 0.6)
        if q > 0:
            C[2] += 900 * q
            if win["layer"] == "front":
                op *= 1 - q
        return C, (rx, ry, rz), op

    def draw_win(self, dst, base, patch, mask, quad, z, blur, mv, zoom_amt, opacity, shadow, glass_kw):
        if quad is None or opacity <= 0.01:
            return
        x0, y0 = np.floor(quad.min(0)).astype(int) - 4
        x1, y1 = np.ceil(quad.max(0)).astype(int) + 4
        x0, y0 = max(x0, -120), max(y0, -120)
        x1, y1 = min(x1, c.W + 120), min(y1, c.H + 120)
        if x1 - x0 < 8 or y1 - y0 < 8:
            return
        h, w = mask.shape
        Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [w, 0], [w, h], [0, h]]),
                                         (quad - np.float32([x0, y0])).astype(np.float32))
        size = (int(x1 - x0), int(y1 - y0))
        m = cv2.warpPerspective(mask, Hm, size, flags=cv2.INTER_LINEAR)
        g = fx.glass(base, m, x0, y0, **glass_kw)
        ct = cv2.warpPerspective(patch, Hm, size, flags=cv2.INTER_LINEAR)
        c.over(g, ct)
        pad = 0
        if blur > 0.6 or zoom_amt > 0.02 or math.hypot(*mv) > 1.5:
            pad = int(3 * blur + max(size) * min(zoom_amt, 0.4) * 0.6 + math.hypot(*mv) / 2) + 6
            g = np.pad(g, ((pad, pad), (pad, pad), (0, 0)))
            if blur > 0.6:
                g = cv2.GaussianBlur(g, (0, 0), blur)
            if zoom_amt > 0.02:
                g = fx.zoom_blur(g, g.shape[1] / 2, g.shape[0] / 2, min(zoom_amt, 0.4))
            g, p2 = fx.blur_patch(g, *mv)
            pad += p2
        if shadow > 0:
            sh, (ox, oy) = c.shadow(g, 30, shadow, 22)
            c.over(dst, sh, x0 - pad + ox, y0 - pad + oy, opacity)
        c.over(dst, g, x0 - pad, y0 - pad, opacity)

    def win_at(self, win, t, cam):
        C, rot, op = self.pose(win, t, cam)
        R = fx.rot3(cam["pitch"], cam["yaw"], 0)
        q, z = self.quad(win["w"], win["h"], C, rot, R, cam["dolly"])
        return q, z, op

    def frame(self, i, rgb, alpha):
        t = i / self.clip.fps
        dt = 0.6 / self.clip.fps
        cam = self.orbit_at(t)
        camp = self.orbit_at(t - dt)
        img, a = self.cam.place(i, t, rgb, alpha)
        img = c.grade(img, contrast=1.07, sat=1.02, warm=0.006)
        plate, _ = self.cam.place(i, t, self.clip.plate, alpha)
        q_pay = fx.smooth(t, self.pay_t - 0.08, 0.6)
        focus = c.lerp(0.0, -300.0, q_pay)

        def dof(z):
            dz = abs(float(np.mean(z)) - PERSP - focus)
            return max(0.0, dz - 230) * 0.0075

        # live preview crop for the preview window
        live = None
        pv = [w for w in self.wins if w["name"] == "preview"][0]
        fb = self.clip.face[min(i, self.clip.n - 1)]
        ks = 0.25
        small = cv2.resize(rgb, (int(rgb.shape[1] * ks), int(rgb.shape[0] * ks)), interpolation=cv2.INTER_AREA)
        live = c.grade(face_crop(small, fb, ks, pv["w"] - 40, 236, span=2.2), contrast=1.08)

        items = []
        for win in self.wins:
            q, z, op = self.win_at(win, t, cam)
            if q is None:
                continue
            qp, zp, _ = self.win_at(win, t - dt, camp)
            mv = (0.0, 0.0)
            zoom_amt = 0.0
            if qp is not None:
                mv = tuple((q.mean(0) - qp.mean(0)).tolist())
                zoom_amt = abs(float(np.mean(zp)) / float(np.mean(z)) - 1) * 1.6
            items.append((float(np.mean(z)), win, q, z, op, mv, zoom_amt))
        items.sort(key=lambda it: -it[0])

        gk_back = dict(bevel=22, refract=1.0, magnify=0.0, frost=5.0, chroma=0.5, tint=(16, 20, 32), tint_a=0.42,
                       spec=0.8, rim=0.7)
        gk_front = dict(bevel=26, refract=1.1, magnify=0.0, frost=4.0, chroma=0.6, tint=(14, 18, 30), tint_a=0.48,
                        spec=0.9, rim=0.8)
        # back windows: behind the speaker (matte sandwich), refracting the clean wall
        back = img.copy()
        for zm, win, q, z, op, mv, za in items:
            if win["layer"] != "back":
                continue
            p = self.content(win, t, i, None)
            self.draw_win(back, plate, p, win["mask"], q, z, dof(z), mv, za, op, 0.0, gk_back)
        img = img * a[..., None] + back * (1 - a[..., None])
        # rack focus onto the payoff window
        if q_pay > 0.01:
            img = cv2.GaussianBlur(img, (0, 0), 0.3 + 5.5 * q_pay) * (1 - 0.18 * q_pay)
        # front windows
        for zm, win, q, z, op, mv, za in items:
            if win["layer"] != "front":
                continue
            p = self.content(win, t, i, live if win["name"] == "preview" else None)
            self.draw_win(img, img, p, win["mask"], q, z, dof(z), mv, za, op, 0.32, gk_front)
            if win["name"] == "prompt":
                self.prompt_geo = (win, q)
        self.payoff(img, t, cam, camp, dof)
        self.draw_cursor(img, t, cam)
        img = c.bloom(img, 205, 24, 0.22)
        img = c.vignette(img, 0.3)
        return c.grain(img, i, 2.4)

    def orbit_at(self, t):
        o = track(self.orbit, t, ("yaw", "pitch", "dolly"))
        o["yaw"] += 2.2 * math.sin(t * 0.9)
        o["pitch"] += 1.0 * math.sin(t * 0.7 + 1.0)
        return o

    # ---- payoff: a glass window holding his freeze-frame flies toward camera ----
    def pay_pose(self, t):
        e = fx.spring(t, self.pay_t - 0.06, 1.7, 0.7)
        C = np.float32([c.lerp(420, 0, e), c.lerp(-900, 300, e), c.lerp(4200, -260, e)])
        rot = (c.lerp(18, 3, e), c.lerp(-50, -5, e), c.lerp(16, -2, e))
        return C, rot, e

    def payoff(self, img, t, cam, camp, dof):
        if t < self.pay_t - 0.06:
            return
        w, h = 600, 820
        if not hasattr(self, "_pay_mask"):
            self._pay_mask = c.rrect(w, h, 40)
        b = self.B[-1]
        p = c.layer(h, w)
        c.over(p, rgb_patch(self.freeze, 26), 24, 24)
        lab = self.tx(b["label"], "grotesk", 66, INK, 780)
        if lab.shape[1] > w - 60:
            lab = self.tx(b["label"], "grotesk", int(66 * (w - 60) / lab.shape[1]), INK, 780)
        sub = b.get("sub", "")
        if sub:
            c.over(p, self.tx(sub, "mono", 28, ACC, 600, 0.16), 30, 646)
        c.over(p, lab, 28, 684)
        # generated badge
        bd = grad_pill(206, 52, ACC, ACC2)
        c.over(bd, sparkle(22), 18, 15)
        c.over(bd, self.tx("Generated", "grotesk", 28, INK, 640), 48, 9)
        c.over(p, bd, 24 + 552 - 206 - 18, 42)
        sw = c.prog(t, self.pay_t + 0.42, 0.75)
        if 0 < sw < 1:
            p = fx.light_sweep(p, sw, width=0.12, angle=24, strength=0.5)
        C, rot, e = self.pay_pose(t)
        R = fx.rot3(cam["pitch"], cam["yaw"], 0)
        q, z = self.quad(w, h, C, rot, R, cam["dolly"])
        if q is None:
            return
        Cp, rotp, _ = self.pay_pose(t - 0.6 / self.clip.fps)
        qp, zp = self.quad(w, h, Cp, rotp, fx.rot3(camp["pitch"], camp["yaw"], 0), camp["dolly"])
        mv, za = (0.0, 0.0), 0.0
        if qp is not None:
            mv = tuple((q.mean(0) - qp.mean(0)).tolist())
            za = abs(float(np.mean(zp)) / float(np.mean(z)) - 1) * 1.6
        # accent glow behind the window as it lands
        gl = c.clamp(e) * 0.5
        if gl > 0.02:
            cx, cy = q.mean(0)
            r = int(max(np.ptp(q[:, 0]), np.ptp(q[:, 1])) * 0.62)
            ys, xs = fx.grid(2 * r, 2 * r)
            m = np.exp(-((xs - r) ** 2 + (ys - r) ** 2) / (2 * (r * 0.45) ** 2)) * gl
            fx.add(img, c.fill(m, ACC), cx - r, cy - r, 0.55)
        self.draw_win(img, img, p, self._pay_mask, q, z, dof(z) * 0.3, mv, za, c.clamp((t - self.pay_t + 0.06) / 0.06),
                      0.45, dict(bevel=30, refract=1.2, magnify=0.0, frost=3.0, chroma=0.7, tint=(14, 18, 30),
                                 tint_a=0.5, spec=1.0, rim=0.9))

    # ---- cursor: glides on an eased bezier, presses Generate ----
    def draw_cursor(self, img, t, cam):
        t0, t1 = self.press_t - 0.62, self.press_t - 0.04
        if t < t0 or not hasattr(self, "prompt_geo"):
            return
        win, q = self.prompt_geo
        bx, by = self.btn_local
        u, v = bx / win["w"], by / win["h"]
        top = q[0] + (q[1] - q[0]) * u
        bot = q[3] + (q[2] - q[3]) * u
        tgt = top + (bot - top) * v + np.float32([10, 8])
        p0 = np.float32([1180, 1880])
        p1 = np.float32([1040, 1180])
        p2 = tgt + np.float32([240, -260])
        s = fx.smooth(t, t0, t1 - t0)

        def bez(s):
            return ((1 - s) ** 3) * p0 + 3 * ((1 - s) ** 2) * s * p1 + 3 * (1 - s) * s * s * p2 + s ** 3 * tgt

        pos = bez(s)
        prev = bez(fx.smooth(t - 0.6 / self.clip.fps, t0, t1 - t0))
        after = fx.expo(t, self.press_t + 0.12, 0.6)
        pos = pos + np.float32([70, 90]) * after
        press = math.sin(math.pi * c.prog(t, self.press_t - 0.05, 0.2))
        op = 1 - fx.smooth(t, self.pay_t - 0.05, 0.25)
        mv = tuple((pos - prev).tolist()) if after == 0 else (0.0, 0.0)
        cp = self.cursor
        fx.put(img, cp, pos[0] + cp.shape[1] / 2 - self.cpad, pos[1] + cp.shape[0] / 2 - self.cpad,
               1 - 0.14 * press, opacity=op, mv=mv, shadow=0.45, sblur=8, sdy=6)
